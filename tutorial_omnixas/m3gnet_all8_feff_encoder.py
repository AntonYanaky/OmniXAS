#!/usr/bin/env python3
"""Train an all-8-FEFF XAS encoder, export 64D features for UniversalXAS.

Modes:
  --init pretrained --supervisor temp_head
  --init pretrained --supervisor frozen_universal
  --init scratch    --supervisor temp_head
  --init scratch    --supervisor frozen_universal  # allowed, usually harder

After this script exports features, train a new UniversalXAS with
`tutorial_omnixas/train_paper_models.py --models universal --data-dir <features>`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import dgl
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pymatgen.core import Lattice, Structure
from torch import nn
from torch.utils.data import DataLoader, Dataset

from matgl import load_model
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph

try:
    from matgl.graph.compute import compute_pair_vector_and_distance, compute_theta_and_phi, create_line_graph
except Exception:  # older MatGL
    from matgl.graph._compute_dgl import compute_pair_vector_and_distance, compute_theta_and_phi, create_line_graph

from matgl.layers import (
    MLP as M3GNetMLP,
    ActivationFunction,
    BondExpansion,
    EmbeddingBlock,
    GatedMLP,
    M3GNetBlock,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
)
from matgl.utils.cutoff import polynomial_cutoff

# MatGL 0.8.x keeps a few helper tensors on CPU unless patched.
import matgl.layers._basis as matgl_basis
import matgl.layers._three_body as matgl_three_body
import matgl.utils.maths as matgl_math

from omnixas.model.xasblock import XASBlock


FEFF_TASKS = ["Ti_FEFF", "V_FEFF", "Cr_FEFF", "Mn_FEFF", "Fe_FEFF", "Co_FEFF", "Ni_FEFF", "Cu_FEFF"]
FEATURE_SCALE = 1000.0
OUTPUT_DIM = 141
UNIVERSAL_DIMS = [500, 500, 550]


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch<2.0
        return torch.load(path, map_location="cpu")


def patch_matgl_gpu_constants() -> None:
    def _call_sbf(self, r):
        cutoff = torch.as_tensor(self.cutoff, dtype=r.dtype, device=r.device)
        r_c = r.clone()
        r_c[r_c > cutoff] = cutoff
        roots = matgl_basis.SPHERICAL_BESSEL_ROOTS[: self.max_l, : self.max_n].to(r.device, dtype=r.dtype)
        factor = torch.sqrt(torch.as_tensor(2.0, dtype=r.dtype, device=r.device) / cutoff**3)
        return torch.cat([
            self.funcs[i](r_c[:, None] * roots[i][None, :] / cutoff)
            * factor
            / torch.abs(self.funcs[i + 1](roots[i][None, :]))
            for i in range(self.max_l)
        ], dim=1)

    def _combine(sbf, shf, max_n: int, max_l: int, use_phi: bool):
        if sbf.size(0) == 0:
            return sbf
        if use_phi:
            repeats = torch.repeat_interleave(2 * torch.arange(max_l, device=sbf.device) + 1, max_n)
            block_size = 2 * torch.arange(max_l, device=sbf.device) + 1
        else:
            repeats = torch.ones(max_l * max_n, dtype=torch.long, device=sbf.device)
            block_size = [1] * max_l
        sbf = torch.repeat_interleave(sbf, repeats, 1)
        cols = torch.arange(shf.size(1), device=shf.device)
        idx, start = [], 0
        for block in block_size:
            block = int(block.item()) if torch.is_tensor(block) else int(block)
            idx.append(torch.tile(cols[start : start + block], [max_n]))
            start += block
        shape = max_n * max_l * (max_l if use_phi else 1)
        return torch.reshape(sbf * torch.index_select(shf, 1, torch.cat(idx)), [-1, shape])

    def _scatter_sum(input_tensor, segment_ids, num_segments: int, dim: int):
        segment_ids = matgl_math.broadcast(segment_ids.to(input_tensor.device), input_tensor, dim)
        size = list(input_tensor.size())
        size[dim] = 0 if segment_ids.numel() == 0 else num_segments
        return torch.zeros(size, dtype=input_tensor.dtype, device=input_tensor.device).scatter_add_(dim, segment_ids, input_tensor)

    matgl_basis.SphericalBesselFunction._call_sbf = _call_sbf
    matgl_basis.combine_sbf_shf = _combine
    matgl_three_body.combine_sbf_shf = _combine
    matgl_math.scatter_sum = _scatter_sum
    matgl_three_body.scatter_sum = _scatter_sum


def read_id_site(path: Path) -> list[tuple[str, int]]:
    return [(mid, int(site)) for line in path.read_text().splitlines() if line.strip() for mid, site in [line.strip().rsplit("_", 1)]]


def parse_feff_structure(path: Path) -> Structure:
    abc = angles = None
    species, coords = [], []
    site_re = re.compile(r"^\*\s+\d+\s+([A-Z][a-z]?)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("TITLE abc:"):
            abc = [float(x) for x in line.split(":", 1)[1].split()[:3]]
        elif line.startswith("TITLE angles:"):
            angles = [float(x) for x in line.split(":", 1)[1].split()[:3]]
        elif match := site_re.match(line):
            species.append(match.group(1))
            coords.append([float(match.group(i)) for i in range(2, 5)])
    if abc is None or angles is None or not species:
        raise ValueError(f"Could not parse FEFF structure: {path}")
    return Structure(Lattice.from_parameters(*abc, *angles), species, coords, coords_are_cartesian=False)


def load_feff_structure(raw_root: Path, task: str, material_id: str, site: int) -> Structure:
    element, kind = task.split("_", 1)
    if kind != "FEFF":
        raise ValueError(f"Only FEFF tasks are supported here, got {task}")
    material_dir = raw_root / "FEFF" / element / material_id
    poscar = material_dir / "POSCAR"
    return Structure.from_file(poscar) if poscar.exists() else parse_feff_structure(material_dir / "FEFF-XANES" / f"{site:03d}_{element}" / "feff.inp")


class FEFFDataset(Dataset):
    def __init__(self, root: Path, raw_root: Path, tasks: list[str], split: str, max_per_task: int | None = None):
        self.root = root
        self.raw_root = raw_root
        self.rows: list[tuple[str, str, int, torch.Tensor, torch.Tensor]] = []
        self.cache: dict[tuple[str, str], Structure] = {}
        data_dir = root / "tutorial_omnixas" / "ml_data"
        id_dir = root / "tutorial_omnixas" / "material_id_and_site"
        for task in tasks:
            ids = read_id_site(id_dir / f"{task}_{split}.txt")
            y = np.atleast_2d(np.loadtxt(data_dir / f"{task}_{split}_y.txt", dtype=np.float32))
            anchor = np.atleast_2d(np.loadtxt(data_dir / f"{task}_{split}_X.txt", dtype=np.float32)) / FEATURE_SCALE
            if max_per_task is not None:
                ids, y, anchor = ids[:max_per_task], y[:max_per_task], anchor[:max_per_task]
            for (mid, site), yi, ai in zip(ids, y, anchor, strict=True):
                self.rows.append((task, mid, site, torch.tensor(yi), torch.tensor(ai)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        task, mid, site, y, anchor = self.rows[idx]
        key = (task, mid)
        if key not in self.cache:
            self.cache[key] = load_feff_structure(self.raw_root, task, mid, site)
        return task, self.cache[key], site, y, anchor


class CollateGraphs:
    def __init__(self, encoder, task_to_idx: dict[str, int]):
        self.converter = Structure2Graph(encoder.element_types, encoder.cutoff)
        self.task_to_idx = task_to_idx

    def graph(self, structure: Structure):
        out = self.converter.get_graph(structure)
        if len(out) == 2:
            graph, _ = out
            lat = torch.as_tensor(structure.lattice.matrix, dtype=torch.float32)
        else:
            graph, lat, _ = out
            lat = torch.as_tensor(lat[0] if getattr(lat, "ndim", 0) == 3 else lat, dtype=torch.float32)
        if "pbc_offshift" in graph.edata:
            graph.edata["pbc_offshift"] = graph.edata["pbc_offshift"].to(torch.float32)
        else:
            graph.edata["pbc_offshift"] = graph.edata["pbc_offset"].to(torch.float32) @ lat
        if "pos" in graph.ndata:
            graph.ndata["pos"] = graph.ndata["pos"].to(torch.float32)
        elif "frac_coords" in graph.ndata:
            graph.ndata["pos"] = graph.ndata["frac_coords"].to(torch.float32) @ lat
        else:
            graph.ndata["pos"] = torch.as_tensor(structure.cart_coords, dtype=torch.float32)
        graph.edata["bond_vec"], graph.edata["bond_dist"] = compute_pair_vector_and_distance(graph)
        return graph

    def __call__(self, batch):
        graphs, sites, tasks, y, anchor = [], [], [], [], []
        offset = 0
        for task, structure, site, yi, ai in batch:
            graph = self.graph(structure)
            graphs.append(graph)
            sites.append(offset + site)
            tasks.append(self.task_to_idx[task])
            y.append(yi)
            anchor.append(ai)
            offset += graph.num_nodes()
        return {
            "graph": dgl.batch(graphs),
            "site": torch.tensor(sites, dtype=torch.long),
            "task": torch.tensor(tasks, dtype=torch.long),
            "y": torch.stack(y).float(),
            "anchor": torch.stack(anchor).float(),
        }


class TempSpectrumHead(nn.Module):
    def __init__(self, hidden_dims: list[int], dropout: float):
        super().__init__()
        dims = [64, *hidden_dims, OUTPUT_DIM]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.extend([nn.BatchNorm1d(b), nn.SiLU(), nn.Dropout(dropout)])
            else:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def infer_xasblock_dims(ckpt_path: Path) -> tuple[int, list[int], int]:
    ckpt = torch_load(ckpt_path)
    state = ckpt.get("state_dict", ckpt)
    layers = []
    for key, value in state.items():
        match = re.fullmatch(r"model\.(\d+)\.weight", key)
        if match and value.ndim == 2:
            layers.append((int(match.group(1)), tuple(value.shape)))
    if not layers:
        raise ValueError(f"Could not infer XASBlock dims from {ckpt_path}")
    layers.sort()
    dims = [layers[0][1][1]] + [shape[0] for _, shape in layers]
    return dims[0], dims[1:-1], dims[-1]


def load_xas_head(ckpt_path: Path) -> XASBlock:
    input_dim, hidden_dims, output_dim = infer_xasblock_dims(ckpt_path)
    if input_dim != 64 or output_dim != OUTPUT_DIM:
        raise ValueError(f"Expected 64->{OUTPUT_DIM} XASBlock, got {input_dim}->{output_dim}: {ckpt_path}")
    head = XASBlock(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    ckpt = torch_load(ckpt_path)
    state = ckpt.get("state_dict", ckpt)
    head.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
    return head


def checkpoint_score(ckpt_path: Path) -> float:
    try:
        ckpt = torch_load(ckpt_path)
        scores = []
        for name, state in ckpt.get("callbacks", {}).items():
            if "ModelCheckpoint" in str(name):
                score = state.get("best_model_score")
                if score is not None:
                    scores.append(float(score.detach().cpu().item() if torch.is_tensor(score) else score))
        if scores:
            return min(scores)
    except Exception:
        pass
    return float("inf")


def find_universal_checkpoint(root: Path, requested: str) -> Path:
    if requested != "latest":
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    patterns = [
        root / "output" / "training" / "universalXAS" / "All_FEFF" / "runs" / "*" / "best*.ckpt",
        root / "output" / "training" / "universalXAS" / "All_FEFF" / "checkpoints" / "best*.ckpt",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(
            "No UniversalXAS checkpoint found. First run: "
            "python tutorial_omnixas/train_paper_models.py --models universal --seed 42"
        )
    scored = sorted((checkpoint_score(path), -path.stat().st_mtime, path) for path in matches)
    best_score, _, best_path = scored[0]
    if not np.isfinite(best_score):
        best_path = sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
    return best_path


class PretrainedM3GNetEncoder(nn.Module):
    def __init__(self, m3gnet, train_block_indices: list[int]):
        super().__init__()
        self.m3gnet = m3gnet
        self.element_types = m3gnet.element_types
        self.cutoff = m3gnet.cutoff
        self.threebody_cutoff = m3gnet.threebody_cutoff
        self.train_block_indices = self._normalize_block_indices(train_block_indices)

        for param in self.m3gnet.parameters():
            param.requires_grad = False
        for idx in self.train_block_indices:
            for param in self.m3gnet.three_body_interactions[idx].parameters():
                param.requires_grad = True
            for param in self.m3gnet.graph_layers[idx].parameters():
                param.requires_grad = True

    def _normalize_block_indices(self, indices: list[int]) -> list[int]:
        out = []
        for idx in indices:
            idx = self.m3gnet.n_blocks + idx if idx < 0 else idx
            if idx < 0 or idx >= self.m3gnet.n_blocks:
                raise ValueError(f"Invalid M3GNet block index {idx}")
            out.append(idx)
        return sorted(set(out))

    def forward(self, graph, site):
        node_types = graph.ndata["node_type"]
        graph.edata["rbf"] = self.m3gnet.bond_expansion(graph.edata["bond_dist"])
        line_graph = create_line_graph(graph.to("cpu"), self.m3gnet.threebody_cutoff).to(graph.device)
        line_graph.apply_edges(compute_theta_and_phi)
        three_body_basis = self.m3gnet.basis_expansion(line_graph)
        three_body_cutoff = polynomial_cutoff(graph.edata["bond_dist"], self.m3gnet.threebody_cutoff)

        with torch.no_grad():
            node, edge, state = self.m3gnet.embedding(node_types, graph.edata["rbf"], None)

        train_set = set(self.train_block_indices)
        for idx in range(self.m3gnet.n_blocks):
            if idx in train_set:
                edge = self.m3gnet.three_body_interactions[idx](graph, line_graph, three_body_basis, three_body_cutoff, node, edge)
                edge, node, state = self.m3gnet.graph_layers[idx](graph, edge, node, state)
            else:
                with torch.no_grad():
                    edge = self.m3gnet.three_body_interactions[idx](graph, line_graph, three_body_basis, three_body_cutoff, node, edge)
                    edge, node, state = self.m3gnet.graph_layers[idx](graph, edge, node, state)
        return node[site]


class ScratchM3GNetEncoder(nn.Module):
    def __init__(self, cutoff: float, threebody_cutoff: float, blocks: int, dropout: float):
        super().__init__()
        act = ActivationFunction["swish"].value()
        max_n = max_l = 3
        degree = max_n * max_l
        self.element_types = DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.blocks = blocks
        self.bond_expansion = BondExpansion(max_l, max_n, cutoff)
        self.basis_expansion = SphericalBesselWithHarmonics(max_n, max_l, cutoff, use_smooth=False, use_phi=False)
        self.embedding = EmbeddingBlock(
            degree_rbf=degree,
            dim_node_embedding=64,
            dim_edge_embedding=64,
            ntypes_node=len(DEFAULT_ELEMENTS),
            activation=act,
        )
        self.three_body_interactions = nn.ModuleList([
            ThreeBodyInteractions(
                update_network_atom=M3GNetMLP(dims=[64, degree], activation=nn.Sigmoid(), activate_last=True),
                update_network_bond=GatedMLP(in_feats=degree, dims=[64], use_bias=False),
            )
            for _ in range(blocks)
        ])
        self.graph_layers = nn.ModuleList([
            M3GNetBlock(
                degree=degree,
                activation=act,
                conv_hiddens=[64, 64],
                dim_node_feats=64,
                dim_edge_feats=64,
                dropout=dropout,
            )
            for _ in range(blocks)
        ])

    def forward(self, graph, site):
        graph.edata["rbf"] = self.bond_expansion(graph.edata["bond_dist"])
        line_graph = create_line_graph(graph.to("cpu"), self.threebody_cutoff).to(graph.device)
        line_graph.apply_edges(compute_theta_and_phi)
        basis = self.basis_expansion(line_graph)
        cutoff = polynomial_cutoff(graph.edata["bond_dist"], self.threebody_cutoff)
        node, edge, state = self.embedding(graph.ndata["node_type"], graph.edata["rbf"], None)
        for idx in range(self.blocks):
            edge = self.three_body_interactions[idx](graph, line_graph, basis, cutoff, node, edge)
            edge, node, state = self.graph_layers[idx](graph, edge, node, state)
        return node[site]


class EncoderSpectrumModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, frozen_head: bool):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.frozen_head = frozen_head
        if frozen_head:
            for param in self.head.parameters():
                param.requires_grad = False
            self.head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_head:
            self.head.eval()
        return self

    def forward(self, graph, site):
        z = self.encoder(graph, site)
        return self.head(z * FEATURE_SCALE), z


class LitEncoder(pl.LightningModule):
    def __init__(
        self,
        model: EncoderSpectrumModel,
        tasks: list[str],
        train_baseline: torch.Tensor,
        val_baseline: torch.Tensor,
        task_weights: torch.Tensor,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
        anchor_lambda: float,
        deriv_lambda: float,
        max_epochs: int,
    ):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.register_buffer("train_baseline", train_baseline.float())
        self.register_buffer("val_baseline", val_baseline.float())
        self.register_buffer("task_weights", task_weights.float())
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.anchor_lambda = anchor_lambda
        self.deriv_lambda = deriv_lambda
        self.max_epochs = max_epochs
        self.val_mses: list[torch.Tensor] = []
        self.val_tasks: list[torch.Tensor] = []

    def step(self, batch, stage: str):
        graph = batch["graph"].to(self.device)
        site = batch["site"].to(self.device)
        task = batch["task"].to(self.device)
        y = batch["y"].to(self.device)
        anchor = batch["anchor"].to(self.device)
        pred, z = self.model(graph, site)

        sample_mse = ((pred - y) ** 2).mean(dim=1)
        rel_mse = sample_mse / self.train_baseline[task].clamp_min(1e-12)
        balanced_rel_mse = (rel_mse * self.task_weights[task]).mean()

        loss = balanced_rel_mse
        if self.anchor_lambda:
            loss = loss + self.anchor_lambda * ((z - anchor) ** 2).mean()
        if self.deriv_lambda:
            deriv = ((torch.diff(pred, dim=1) - torch.diff(y, dim=1)) ** 2).mean()
            loss = loss + self.deriv_lambda * deriv

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mse", sample_mse.mean(), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_balanced_rel_mse", balanced_rel_mse, on_epoch=True, prog_bar=True)
        if stage == "val":
            self.val_mses.append(sample_mse.detach())
            self.val_tasks.append(task.detach())
        return loss

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def on_validation_epoch_start(self):
        self.val_mses = []
        self.val_tasks = []

    def validation_step(self, batch, _):
        return self.step(batch, "val")

    def on_validation_epoch_end(self):
        if not self.val_mses:
            return
        mses = torch.cat(self.val_mses)
        tasks = torch.cat(self.val_tasks)
        rel_medians = []
        for idx, task_name in enumerate(self.tasks):
            mask = tasks == idx
            if not mask.any():
                continue
            med = mses[mask].median()
            eta = self.val_baseline[idx] / med.clamp_min(1e-12)
            rel_medians.append(med / self.val_baseline[idx].clamp_min(1e-12))
            self.log(f"val_median_mse_{task_name}", med, on_epoch=True)
            self.log(f"val_eta_{task_name}", eta, on_epoch=True, prog_bar=(task_name == "Cu_FEFF"))
        if rel_medians:
            self.log("val_balanced_rel_mse", torch.stack(rel_medians).mean(), on_epoch=True, prog_bar=True)
            self.log("val_mean_eta", torch.stack([1 / r.clamp_min(1e-12) for r in rel_medians]).mean(), on_epoch=True)

    def configure_optimizers(self):
        groups = []
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.encoder_lr, "name": "encoder"})
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        if head_params:
            groups.append({"params": head_params, "lr": self.head_lr, "name": "head"})
        if not groups:
            raise RuntimeError("No trainable parameters selected")
        opt = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-6),
                "interval": "epoch",
            },
        }


def read_y(root: Path, task: str, split: str) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(root / "tutorial_omnixas" / "ml_data" / f"{task}_{split}_y.txt", dtype=np.float32))


def baseline_medians(root: Path, tasks: list[str], split: str) -> torch.Tensor:
    out = []
    for task in tasks:
        train_y = read_y(root, task, "train")
        target = read_y(root, task, split)
        baseline = np.repeat(train_y.mean(axis=0, keepdims=True), len(target), axis=0)
        out.append(float(np.median(np.mean((target - baseline) ** 2, axis=1))))
    return torch.tensor(out, dtype=torch.float32)


def task_counts(root: Path, tasks: list[str]) -> torch.Tensor:
    counts = [len(read_y(root, task, "train")) for task in tasks]
    return torch.tensor(counts, dtype=torch.float32)


def build_model(args, root: Path) -> tuple[EncoderSpectrumModel, Path | None]:
    universal_ckpt = None
    if args.init == "pretrained":
        base = load_model(str(root / "models" / "M3GNet-MP-2021.2.8-PES")).model.eval()
        encoder = PretrainedM3GNetEncoder(base, args.train_blocks)
    elif args.init == "scratch":
        encoder = ScratchM3GNetEncoder(args.cutoff, args.threebody_cutoff, args.blocks, args.gnn_dropout)
    else:
        raise ValueError(args.init)

    if args.supervisor == "temp_head":
        head = TempSpectrumHead(args.temp_head_dims, args.temp_head_dropout)
        frozen_head = False
    elif args.supervisor == "frozen_universal":
        universal_ckpt = find_universal_checkpoint(root, args.universal_ckpt)
        head = load_xas_head(universal_ckpt)
        frozen_head = True
    else:
        raise ValueError(args.supervisor)

    return EncoderSpectrumModel(encoder, head, frozen_head), universal_ckpt


def load_model_from_checkpoint(args, root: Path, ckpt_path: Path) -> EncoderSpectrumModel:
    model, _ = build_model(args, root)
    ckpt = torch_load(ckpt_path)
    state = ckpt.get("state_dict", ckpt)
    model_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=True)
    return model.eval()


@torch.no_grad()
def export_features(
    model: EncoderSpectrumModel,
    root: Path,
    raw_root: Path,
    tasks: list[str],
    out_dir: Path,
    batch_size: int,
    num_workers: int,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    collate = CollateGraphs(model.encoder, task_to_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for split in ["train", "val", "test"]:
            ds = FEFFDataset(root, raw_root, [task], split)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
            feats, targets = [], []
            for batch in loader:
                z = model.encoder(batch["graph"].to(device), batch["site"].to(device))
                feats.append((z * FEATURE_SCALE).cpu().numpy())
                targets.append(batch["y"].numpy())
            X = np.concatenate(feats)
            y = np.concatenate(targets)
            np.savetxt(out_dir / f"{task}_{split}_X.txt", X)
            np.savetxt(out_dir / f"{task}_{split}_y.txt", y)
            print(f"exported {task} {split}: {X.shape} {y.shape}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    parser.add_argument("--supervisor", choices=["temp_head", "frozen_universal"], default="temp_head")
    parser.add_argument("--tasks", nargs="+", default=FEFF_TASKS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--max-train-per-task", type=int, default=None)
    parser.add_argument("--max-val-per-task", type=int, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--anchor-lambda", type=float, default=None)
    parser.add_argument("--deriv-lambda", type=float, default=0.02)
    parser.add_argument("--monitor", default="val_balanced_rel_mse")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-blocks", nargs="+", type=int, default=[2], help="Pretrained M3GNet block indices to train. Use '1 2' for last two.")
    parser.add_argument("--universal-ckpt", default="latest", help="For --supervisor frozen_universal: checkpoint path or 'latest'.")
    parser.add_argument("--temp-head-dims", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--temp-head-dropout", type=float, default=0.1)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--threebody-cutoff", type=float, default=4.0)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="output/training/m3gnetAll8FEFF")
    parser.add_argument("--export-dir", default=None, help="Default: <run_dir>/features")
    parser.add_argument("--skip-train", action="store_true", help="Only export from --checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to export when --skip-train, or explicit checkpoint after training.")
    parser.add_argument("--progress-bar", action="store_true", help="Enable Lightning progress bars. Off by default for clean logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_matgl_gpu_constants()
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    raw_root = Path(os.environ.get("OMNIXAS_DATA_ROOT", root.parent / "OmniXAS_data")) / "materialscloud_omnixas_raw" / "extracted"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw data root: {raw_root}. Set OMNIXAS_DATA_ROOT to the directory containing materialscloud_omnixas_raw/.")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    run_label = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{args.init}_{args.supervisor}_seed{args.seed}"
    run_dir = output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=bool(args.skip_train))
    export_dir = Path(args.export_dir) if args.export_dir else run_dir / "features"
    if not export_dir.is_absolute():
        export_dir = root / export_dir

    args.encoder_lr = args.encoder_lr if args.encoder_lr is not None else (2e-6 if args.init == "pretrained" else 1e-3)
    args.head_lr = args.head_lr if args.head_lr is not None else (1e-3 if args.supervisor == "temp_head" else 0.0)
    args.anchor_lambda = args.anchor_lambda if args.anchor_lambda is not None else (1e-4 if args.init == "pretrained" else 0.0)

    ckpt_to_export = Path(args.checkpoint) if args.checkpoint else None
    universal_ckpt = None

    if not args.skip_train:
        model, universal_ckpt = build_model(args, root)
        train_ds = FEFFDataset(root, raw_root, args.tasks, "train", args.max_train_per_task)
        val_ds = FEFFDataset(root, raw_root, args.tasks, "val", args.max_val_per_task)
        task_to_idx = {task: idx for idx, task in enumerate(args.tasks)}
        collate = CollateGraphs(model.encoder, task_to_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.num_workers,
            drop_last=(args.supervisor == "temp_head"),
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

        counts = task_counts(root, args.tasks)
        task_weights = counts.sum() / (len(args.tasks) * counts)
        train_baseline = baseline_medians(root, args.tasks, "train")
        val_baseline = baseline_medians(root, args.tasks, "val")

        metadata = {
            "args": vars(args),
            "tasks": args.tasks,
            "raw_root": str(raw_root),
            "run_dir": str(run_dir),
            "export_dir": str(export_dir),
            "universal_ckpt": str(universal_ckpt) if universal_ckpt else None,
            "train_counts": dict(zip(args.tasks, [int(x) for x in counts.tolist()])),
            "train_baseline_median_mse": dict(zip(args.tasks, [float(x) for x in train_baseline.tolist()])),
            "val_baseline_median_mse": dict(zip(args.tasks, [float(x) for x in val_baseline.tolist()])),
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("run_dir:", run_dir, flush=True)
        print("raw_root:", raw_root, flush=True)
        print("tasks:", args.tasks, flush=True)
        print("train samples:", len(train_ds), "val samples:", len(val_ds), flush=True)
        print("init/supervisor:", args.init, args.supervisor, flush=True)
        print("encoder_lr/head_lr:", args.encoder_lr, args.head_lr, flush=True)
        print("anchor_lambda/deriv_lambda:", args.anchor_lambda, args.deriv_lambda, flush=True)
        print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

        lit = LitEncoder(
            model=model,
            tasks=args.tasks,
            train_baseline=train_baseline,
            val_baseline=val_baseline,
            task_weights=task_weights,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            anchor_lambda=args.anchor_lambda,
            deriv_lambda=args.deriv_lambda,
            max_epochs=args.epochs,
        )
        mode = "max" if "eta" in args.monitor else "min"
        checkpoint = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best-{epoch:03d}-{" + args.monitor + ":.5f}",
            monitor=args.monitor,
            mode=mode,
            save_top_k=1,
            save_last=True,
        )
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator,
            devices=1,
            callbacks=[EarlyStopping(monitor=args.monitor, patience=args.patience, mode=mode), checkpoint],
            logger=CSVLogger(save_dir=str(run_dir), name="logs"),
            log_every_n_steps=1,
            enable_progress_bar=args.progress_bar,
        )
        trainer.fit(lit, train_loader, val_loader)
        if not checkpoint.best_model_path:
            raise RuntimeError("Training finished without a best checkpoint")
        ckpt_to_export = Path(checkpoint.best_model_path)
        shutil.copy2(ckpt_to_export, run_dir / "best_all8_feff_encoder.ckpt")
        print("best checkpoint:", ckpt_to_export, flush=True)

    if ckpt_to_export is None:
        raise ValueError("Need --checkpoint when --skip-train is set")
    trained = load_model_from_checkpoint(args, root, ckpt_to_export)
    export_features(trained, root, raw_root, args.tasks, export_dir, args.batch_size, args.num_workers)
    print("features:", export_dir, flush=True)
    print("next:", flush=True)
    print(
        "python tutorial_omnixas/train_paper_models.py --models universal --seed "
        f"{args.seed} --data-dir {export_dir} --training-root {run_dir / 'heads'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
