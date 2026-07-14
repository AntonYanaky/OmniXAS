#!/usr/bin/env python3
"""Train scratch all-8 FEFF encoder, UniversalXAS, and tuned UniversalXAS."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu

import dgl
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pymatgen.core import Lattice, Structure
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph

try:
    from matgl.graph.compute import compute_pair_vector_and_distance, compute_theta_and_phi, create_line_graph
except Exception:
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

import matgl.layers._basis as matgl_basis
import matgl.layers._three_body as matgl_three_body
import matgl.utils.maths as matgl_math

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.metrics import ModelMetrics
from omnixas.model.training import PlModule
from omnixas.model.xasblock import XASBlock
from omnixas.model.xasblock_regressor import XASBlockRegressor

FEFF_TASKS = ["Ti_FEFF", "V_FEFF", "Cr_FEFF", "Mn_FEFF", "Fe_FEFF", "Co_FEFF", "Ni_FEFF", "Cu_FEFF"]
VASP_TASKS = ["Ti_VASP", "Cu_VASP"]
EXPORT_TASKS = FEFF_TASKS + VASP_TASKS
ELEMENTS = [task.split("_")[0] for task in FEFF_TASKS]
SPLITS = ["train", "val", "test"]
BATCH = {"Ti_FEFF": 32, "V_FEFF": 32, "Cr_FEFF": 32, "Mn_FEFF": 64, "Fe_FEFF": 64, "Co_FEFF": 32, "Ni_FEFF": 32, "Cu_FEFF": 32, "Ti_VASP": 64, "Cu_VASP": 64}
FEATURE_SCALE = 1000.0
INPUT_DIM = 64
OUTPUT_DIM = 141
HEAD_WIDTHS = [500, 500, 550]
TEMP_HEAD_WIDTHS = [256, 256]

ENCODER_BATCH = 24
ENCODER_PATIENCE = 30
ENCODER_HEAD_LR = 1e-3
ENCODER_WEIGHT_DECAY = 1e-5
ENCODER_DERIV_LAMBDA = 0.02
ENCODER_BLOCKS = 3
ENCODER_CUTOFF = 4.0
ENCODER_THREEBODY_CUTOFF = 4.0
ENCODER_TEMP_DROPOUT = 0.1
HEAD_MONITOR = "val_median_mse"


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def patch_matgl_gpu_constants() -> None:
    def _call_sbf(self, r):
        cutoff = torch.as_tensor(self.cutoff, dtype=r.dtype, device=r.device)
        roots = matgl_basis.SPHERICAL_BESSEL_ROOTS[: self.max_l, : self.max_n].to(r.device, dtype=r.dtype)
        factor = torch.sqrt(torch.as_tensor(2.0, dtype=r.dtype, device=r.device) / cutoff**3)
        r = r.clamp(max=cutoff)
        return torch.cat([
            self.funcs[i](r[:, None] * roots[i][None, :] / cutoff) * factor / torch.abs(self.funcs[i + 1](roots[i][None, :]))
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
        cols = torch.arange(shf.size(1), device=shf.device)
        idx, start = [], 0
        for block in block_size:
            block = int(block.item()) if torch.is_tensor(block) else int(block)
            idx.append(torch.tile(cols[start : start + block], [max_n]))
            start += block
        sbf = torch.repeat_interleave(sbf, repeats, 1)
        return torch.reshape(sbf * torch.index_select(shf, 1, torch.cat(idx)), [-1, max_n * max_l * (max_l if use_phi else 1)])

    def _scatter_sum(x, segment_ids, num_segments: int, dim: int):
        segment_ids = matgl_math.broadcast(segment_ids.to(x.device), x, dim)
        size = list(x.size())
        size[dim] = 0 if segment_ids.numel() == 0 else num_segments
        return torch.zeros(size, dtype=x.dtype, device=x.device).scatter_add_(dim, segment_ids, x)

    matgl_basis.SphericalBesselFunction._call_sbf = _call_sbf
    matgl_basis.combine_sbf_shf = _combine
    matgl_three_body.combine_sbf_shf = _combine
    matgl_math.scatter_sum = _scatter_sum
    matgl_three_body.scatter_sum = _scatter_sum


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


def load_structure(raw_root: Path, task: str, material_id: str, site: int) -> Structure:
    element, kind = task.split("_", 1)
    material_dir = raw_root / kind / element / material_id
    if kind == "VASP":
        return Structure.from_file(material_dir / "VASP" / f"{site:03d}_{element}" / "POSCAR")
    poscar = material_dir / "POSCAR"
    if poscar.exists():
        return Structure.from_file(poscar)
    return parse_feff_structure(material_dir / "FEFF-XANES" / f"{site:03d}_{element}" / "feff.inp")


class FEFFDataset(Dataset):
    def __init__(self, root: Path, raw_root: Path, tasks: list[str], split: str):
        self.raw_root = raw_root
        self.rows = []
        self.cache = {}
        data_dir = root / "tutorial_omnixas" / "ml_data"
        id_dir = root / "tutorial_omnixas" / "material_id_and_site"
        for task in tasks:
            ids = [line.strip().rsplit("_", 1) for line in (id_dir / f"{task}_{split}.txt").read_text().splitlines() if line.strip()]
            y = np.atleast_2d(np.loadtxt(data_dir / f"{task}_{split}_y.txt", dtype=np.float32))
            if len(ids) != len(y):
                raise ValueError(f"Split length mismatch for {task} {split}: ids={len(ids)} y={len(y)}")
            self.rows += [(task, mid, int(site), torch.as_tensor(yi, dtype=torch.float32)) for (mid, site), yi in zip(ids, y, strict=True)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        task, mid, site, y = self.rows[idx]
        key = (task, mid)
        if key not in self.cache:
            self.cache[key] = load_structure(self.raw_root, task, mid, site)
        return task, self.cache[key], site, y


class CollateGraphs:
    def __init__(self, encoder):
        self.converter = Structure2Graph(encoder.element_types, encoder.cutoff)
        self.task_idx = {task: i for i, task in enumerate(EXPORT_TASKS)}

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
        graphs, sites, tasks, y = [], [], [], []
        offset = 0
        for task, structure, site, yi in batch:
            graph = self.graph(structure)
            graphs.append(graph)
            sites.append(offset + site)
            tasks.append(self.task_idx[task])
            y.append(yi)
            offset += graph.num_nodes()
        return {"graph": dgl.batch(graphs), "site": torch.tensor(sites), "task": torch.tensor(tasks), "y": torch.stack(y).float()}


class ScratchEncoder(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        act = ActivationFunction["swish"].value()
        degree = 9
        self.element_types = DEFAULT_ELEMENTS
        self.cutoff = ENCODER_CUTOFF
        self.threebody_cutoff = ENCODER_THREEBODY_CUTOFF
        self.bond_expansion = BondExpansion(3, 3, ENCODER_CUTOFF)
        self.basis_expansion = SphericalBesselWithHarmonics(3, 3, ENCODER_CUTOFF, use_smooth=False, use_phi=False)
        self.embedding = EmbeddingBlock(degree_rbf=degree, dim_node_embedding=INPUT_DIM, dim_edge_embedding=INPUT_DIM, ntypes_node=len(DEFAULT_ELEMENTS), activation=act)
        self.three_body_interactions = nn.ModuleList([
            ThreeBodyInteractions(
                update_network_atom=M3GNetMLP(dims=[INPUT_DIM, degree], activation=nn.Sigmoid(), activate_last=True),
                update_network_bond=GatedMLP(in_feats=degree, dims=[INPUT_DIM], use_bias=False),
            )
            for _ in range(ENCODER_BLOCKS)
        ])
        self.graph_layers = nn.ModuleList([
            M3GNetBlock(degree=degree, activation=act, conv_hiddens=[INPUT_DIM, INPUT_DIM], dim_node_feats=INPUT_DIM, dim_edge_feats=INPUT_DIM, dropout=dropout)
            for _ in range(ENCODER_BLOCKS)
        ])

    def forward(self, graph, site):
        graph.edata["rbf"] = self.bond_expansion(graph.edata["bond_dist"])
        line_graph = create_line_graph(graph.to("cpu"), self.threebody_cutoff).to(graph.device)
        line_graph.apply_edges(compute_theta_and_phi)
        basis = self.basis_expansion(line_graph)
        cutoff = polynomial_cutoff(graph.edata["bond_dist"], self.threebody_cutoff)
        node, edge, state = self.embedding(graph.ndata["node_type"], graph.edata["rbf"], None)
        for idx in range(ENCODER_BLOCKS):
            edge = self.three_body_interactions[idx](graph, line_graph, basis, cutoff, node, edge)
            edge, node, state = self.graph_layers[idx](graph, edge, node, state)
        return node[site]


class EncoderModel(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.encoder = ScratchEncoder(dropout)
        XASBlock.DROPOUT = ENCODER_TEMP_DROPOUT
        self.head = XASBlock(INPUT_DIM, TEMP_HEAD_WIDTHS, OUTPUT_DIM)

    def forward(self, graph, site):
        return self.head(self.encoder(graph, site) * FEATURE_SCALE)


class LitEncoder(pl.LightningModule):
    def __init__(self, model, train_base, val_base, weights, lr, epochs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.register_buffer("train_base", train_base.float())
        self.register_buffer("val_base", val_base.float())
        self.register_buffer("weights", weights.float())
        self.val_mses, self.val_tasks = [], []

    def step(self, batch, stage: str):
        graph, site, task, y = batch["graph"].to(self.device), batch["site"].to(self.device), batch["task"].to(self.device), batch["y"].to(self.device)
        pred = self.model(graph, site)
        sample_mse = ((pred - y) ** 2).mean(dim=1)
        base = self.train_base if stage == "train" else self.val_base
        loss = ((sample_mse / base[task].clamp_min(1e-12)) * self.weights[task]).mean()
        loss = loss + ENCODER_DERIV_LAMBDA * ((torch.diff(pred, dim=1) - torch.diff(y, dim=1)) ** 2).mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mse", sample_mse.mean(), on_epoch=True, prog_bar=True)
        if stage == "val":
            self.val_mses.append(sample_mse.detach())
            self.val_tasks.append(task.detach())
        return loss

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def on_validation_epoch_start(self):
        self.val_mses, self.val_tasks = [], []

    def validation_step(self, batch, _):
        return self.step(batch, "val")

    def on_validation_epoch_end(self):
        if not self.val_mses:
            return
        mses, tasks, rel = torch.cat(self.val_mses), torch.cat(self.val_tasks), []
        for i, task in enumerate(FEFF_TASKS):
            if (mask := tasks == i).any():
                med = mses[mask].median()
                rel.append(med / self.val_base[i].clamp_min(1e-12))
                self.log(f"val_eta_{task}", self.val_base[i] / med.clamp_min(1e-12), on_epoch=True, prog_bar=(task == "Cu_FEFF"))
        if rel:
            self.log("val_balanced_rel_mse", torch.stack(rel).mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [{"params": self.model.encoder.parameters(), "lr": self.lr}, {"params": self.model.head.parameters(), "lr": ENCODER_HEAD_LR}],
            weight_decay=ENCODER_WEIGHT_DECAY,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6), "interval": "epoch"}}


def load_feature_split(features: Path, task: str) -> MLSplits:
    return MLSplits(**{
        split: MLData(
            X=np.atleast_2d(np.loadtxt(features / f"{task}_{split}_X.txt", dtype=np.float32)),
            y=np.atleast_2d(np.loadtxt(features / f"{task}_{split}_y.txt", dtype=np.float32)),
        )
        for split in SPLITS
    })


def regressor(directory: Path, lr: float, dropout: float, batch: int, epochs: int, patience: int) -> XASBlockRegressor:
    XASBlock.DROPOUT = dropout
    return XASBlockRegressor(
        directory=str(directory),
        overwrite_save_dir=False,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=HEAD_WIDTHS,
        batch_size=batch,
        max_epochs=epochs,
        early_stopping_patience=patience,
        initial_lr=lr,
        min_lr=1e-4,
        use_lr_finder=False,
        use_early_stopping=True,
        monitor_metric=HEAD_MONITOR,
        shuffle=True,
        lr_scheduler="cosine",
        cosine_t_max=epochs,
        cosine_eta_min=1e-6,
    )


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", path, flush=True)


def checkpoint_score(ckpt: Path) -> float:
    scores = []
    for name, state in torch_load(ckpt).get("callbacks", {}).items():
        if "ModelCheckpoint" in str(name) and state.get("best_model_score") is not None:
            score = state["best_model_score"]
            scores.append(float(score.detach().cpu().item() if torch.is_tensor(score) else score))
    if scores:
        return min(scores)
    match = re.search(r"val_(?:loss|median_mse)[=_](\d+(?:\.\d+)?)", ckpt.name)
    return float(match.group(1)) if match else float("inf")


def eval_ckpt(ckpt: Path, split: MLSplits, split_name: str) -> dict[str, float]:
    module = PlModule.load_from_checkpoint(checkpoint_path=str(ckpt), model=XASBlock(INPUT_DIM, HEAD_WIDTHS, OUTPUT_DIM), lr=1e-4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    X, y = getattr(split, split_name).X, getattr(split, split_name).y
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=1024):
            preds.append(module(xb.to(device)).detach().cpu().numpy())
    pred = np.concatenate(preds)
    median_mse = float(ModelMetrics(predictions=pred, targets=y).median_of_mse_per_spectra)
    baseline = np.repeat(split.train.y.mean(axis=0, keepdims=True), len(y), axis=0)
    baseline_mse = float(np.median(np.mean((y - baseline) ** 2, axis=1)))
    return {f"{split_name}_median_mse": median_mse, f"{split_name}_baseline_median_mse": baseline_mse, f"{split_name}_eta": baseline_mse / median_mse}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default="output/training/m3gnetAll8FEFF")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default=None, help="GPU id to expose, e.g. 0 or 1. Equivalent to CUDA_VISIBLE_DEVICES.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--gnn-dropout", type=float, default=0.1)
    p.add_argument("--encoder-epochs", type=int, default=300)
    p.add_argument("--universal-lr", type=float, default=7e-4)
    p.add_argument("--universal-dropout", type=float, default=0.25)
    p.add_argument("--universal-epochs", type=int, default=800)
    p.add_argument("--universal-patience", type=int, default=60)
    p.add_argument("--tuned-lr", type=float, default=1e-4)
    p.add_argument("--tuned-dropouts", nargs="+", type=float, default=[0.5, 0.0])
    p.add_argument("--tuned-epochs", type=int, default=1000)
    p.add_argument("--tuned-patience", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(f"NumPy {np.__version__} is incompatible with torch==2.1/MatGL graph conversion. Run: pip install 'numpy<2' --force-reinstall")
    patch_matgl_gpu_constants()
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    root = Path(__file__).resolve().parents[1]
    raw_root = Path(os.environ.get("OMNIXAS_DATA_ROOT", root.parent / "OmniXAS_data")) / "materialscloud_omnixas_raw" / "extracted"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw data root: {raw_root}")
    output_root = Path(args.output_root)
    output_root = output_root if output_root.is_absolute() else root / output_root
    run = output_root / (args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_scratch_all8_seed{args.seed}")
    features, heads = run / "features", run / "heads"
    if args.overwrite and run.exists():
        shutil.rmtree(run)
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}; use --overwrite")
    run.mkdir(parents=True)

    settings = vars(args) | {"run_dir": str(run)}
    (run / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps(settings, indent=2), flush=True)

    model = EncoderModel(args.gnn_dropout)
    collate = CollateGraphs(model.encoder)
    train_loader = DataLoader(FEFFDataset(root, raw_root, FEFF_TASKS, "train"), batch_size=ENCODER_BATCH, shuffle=True, collate_fn=collate, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(FEFFDataset(root, raw_root, FEFF_TASKS, "val"), batch_size=ENCODER_BATCH, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    data_dir = root / "tutorial_omnixas" / "ml_data"
    train_base, val_base, counts = [], [], []
    for task in FEFF_TASKS:
        train_y = np.atleast_2d(np.loadtxt(data_dir / f"{task}_train_y.txt", dtype=np.float32))
        counts.append(len(train_y))
        for split_name, out in [("train", train_base), ("val", val_base)]:
            y = np.atleast_2d(np.loadtxt(data_dir / f"{task}_{split_name}_y.txt", dtype=np.float32))
            baseline = np.repeat(train_y.mean(axis=0, keepdims=True), len(y), axis=0)
            out.append(float(np.median(np.mean((y - baseline) ** 2, axis=1))))
    counts = torch.tensor(counts, dtype=torch.float32)

    lit = LitEncoder(
        model,
        torch.tensor(train_base, dtype=torch.float32),
        torch.tensor(val_base, dtype=torch.float32),
        counts.sum() / (len(FEFF_TASKS) * counts),
        args.encoder_lr,
        args.encoder_epochs,
    )
    encoder_ckpt = ModelCheckpoint(dirpath=run / "checkpoints", filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}", monitor="val_balanced_rel_mse", mode="min", save_top_k=1, save_last=True)
    pl.Trainer(
        max_epochs=args.encoder_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_balanced_rel_mse", patience=ENCODER_PATIENCE, mode="min"), encoder_ckpt],
        logger=CSVLogger(save_dir=str(run), name="logs"),
        log_every_n_steps=1,
    ).fit(lit, train_loader, val_loader)
    if not encoder_ckpt.best_model_path:
        raise RuntimeError("Encoder training finished without a best checkpoint")
    best_encoder = Path(encoder_ckpt.best_model_path)
    shutil.copy2(best_encoder, run / "best_all8_feff_encoder.ckpt")

    model = EncoderModel(args.gnn_dropout)
    state = torch_load(best_encoder).get("state_dict")
    if state is None:
        raise ValueError(f"Checkpoint is missing state_dict: {best_encoder}")
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    collate = CollateGraphs(model.encoder)
    features.mkdir(parents=True)
    with torch.no_grad():
        for task in EXPORT_TASKS:
            for split_name in SPLITS:
                loader = DataLoader(FEFFDataset(root, raw_root, [task], split_name), batch_size=ENCODER_BATCH, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
                Xs, ys = [], []
                for batch in loader:
                    z = model.encoder(batch["graph"].to(device), batch["site"].to(device))
                    Xs.append((z * FEATURE_SCALE).cpu().numpy())
                    ys.append(batch["y"].numpy())
                np.savetxt(features / f"{task}_{split_name}_X.txt", np.concatenate(Xs))
                np.savetxt(features / f"{task}_{split_name}_y.txt", np.concatenate(ys))
                print(f"exported {task} {split_name}", flush=True)

    all_splits = [load_feature_split(features, task) for task in FEFF_TASKS]
    universal_split = MLSplits(
        train=MLData(X=np.concatenate([s.train.X for s in all_splits]), y=np.concatenate([s.train.y for s in all_splits])),
        val=MLData(X=np.concatenate([s.val.X for s in all_splits]), y=np.concatenate([s.val.y for s in all_splits])),
        test=MLData(X=np.concatenate([s.test.X for s in all_splits]), y=np.concatenate([s.test.y for s in all_splits])),
    )
    universal_dir = heads / "universalXAS" / "All_FEFF" / "runs" / f"universal_seed{args.seed}_lr{args.universal_lr}_dropout{args.universal_dropout}"
    universal = regressor(universal_dir, args.universal_lr, args.universal_dropout, 32, args.universal_epochs, args.universal_patience)
    universal.fit(universal_split)
    universal_ckpt = Path(universal.cfg.fetch_checkpoint("best"))

    universal_rows = []
    for task in EXPORT_TASKS:
        element, kind = task.split("_", 1)
        split = load_feature_split(features, task)
        row = {"element": element, "type": kind, "dataset": task, "checkpoint": str(universal_ckpt), "val_loss_score": checkpoint_score(universal_ckpt)}
        row.update(eval_ckpt(universal_ckpt, split, "val"))
        row.update(eval_ckpt(universal_ckpt, split, "test"))
        universal_rows.append(row)
        print(f"universal {task}: val_eta={row['val_eta']:.4f} test_eta={row['test_eta']:.4f}", flush=True)
    csv_write(run / "universal_eval.csv", universal_rows)

    tuned_rows = []
    source_dir = universal_ckpt.parent
    for task in EXPORT_TASKS:
        element, kind = task.split("_", 1)
        split = load_feature_split(features, task)
        for dropout in args.tuned_dropouts:
            out_dir = heads / "tunedUniversalXAS" / task / "runs" / f"tuned_seed{args.seed}_lr{args.tuned_lr}_dropout{dropout}"
            tuned = regressor(source_dir, args.tuned_lr, dropout, BATCH[task], args.tuned_epochs, args.tuned_patience)
            tuned.load("best")
            tuned.cfg.directory = str(out_dir)
            out_dir.mkdir(parents=True, exist_ok=False)
            tuned.fit(split)
            ckpt = Path(tuned.cfg.fetch_checkpoint("best"))
            row = {"element": element, "type": kind, "dataset": task, "dropout": dropout, "checkpoint": str(ckpt), "val_loss_score": checkpoint_score(ckpt)}
            row.update(eval_ckpt(ckpt, split, "val"))
            row.update(eval_ckpt(ckpt, split, "test"))
            tuned_rows.append(row)
            print(f"tuned {task} dropout={dropout}: val_eta={row['val_eta']:.4f} test_eta={row['test_eta']:.4f}", flush=True)
    csv_write(run / "tuned_eval_all.csv", tuned_rows)
    csv_write(run / "tuned_eval.csv", [max([row for row in tuned_rows if row["dataset"] == task], key=lambda row: row["val_eta"]) for task in EXPORT_TASKS])
    print("done:", run, flush=True)


if __name__ == "__main__":
    main()
