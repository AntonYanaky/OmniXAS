"""FEFF structure, graph, and balanced sampling utilities for XAS training."""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
import dgl
import numpy as np
import torch
from matgl.ext.pymatgen import Structure2Graph
try:
    from matgl.graph.compute import compute_pair_vector_and_distance
except ImportError:
    from matgl.graph._compute_dgl import compute_pair_vector_and_distance
from pymatgen.core import Lattice, Structure
from torch.utils.data import Dataset, Sampler

from omnixas.data.ml_data import MLData, MLSplits

FEFF_TASKS = ["Ti_FEFF", "V_FEFF", "Cr_FEFF", "Mn_FEFF", "Fe_FEFF", "Co_FEFF", "Ni_FEFF", "Cu_FEFF"]
SPLITS = ("train", "val", "test")
ENCODER_BATCH = 24


def patch_matgl_gpu_constants() -> None:
    """Patch MatGL 0.8.5 tensor constants so graph operations follow the input device."""
    import matgl.layers._basis as matgl_basis
    import matgl.layers._three_body as matgl_three_body
    import matgl.utils.maths as matgl_math

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
        indices, start = [], 0
        for block in block_size:
            size = int(block.item()) if torch.is_tensor(block) else int(block)
            indices.append(torch.tile(cols[start : start + size], [max_n]))
            start += size
        sbf = torch.repeat_interleave(sbf, repeats, 1)
        return torch.reshape(sbf * torch.index_select(shf, 1, torch.cat(indices)), [-1, max_n * max_l * (max_l if use_phi else 1)])

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


def structure_path(raw_root: Path, task: str, material_id: str, site: int) -> Path:
    element, kind = task.split("_", 1)
    material_dir = raw_root / kind / element / material_id
    poscar = material_dir / "POSCAR"
    if poscar.exists():
        return poscar
    return material_dir / "FEFF-XANES" / f"{site:03d}_{element}" / "feff.inp"


def validate_raw_structures(root: Path, raw_root: Path, tasks: list[str]) -> None:
    missing = []
    id_dir = root / "tutorial_omnixas" / "material_id_and_site"
    for task in tasks:
        for split in SPLITS:
            id_path = id_dir / f"{task}_{split}.txt"
            if not id_path.is_file():
                raise FileNotFoundError(f"Missing identifier file: {id_path}")
            for line in id_path.read_text().splitlines():
                if line.strip():
                    material_id, site = line.strip().rsplit("_", 1)
                    path = structure_path(raw_root, task, material_id, int(site))
                    if not path.exists():
                        missing.append((task, split, line.strip(), path))
    if missing:
        examples = "\n".join(f"  {task} {split} {row}: {path}" for task, split, row, path in missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} raw structure file(s) under {raw_root}.\nFirst missing files:\n{examples}")


class FEFFDataset(Dataset):
    def __init__(self, root: Path, raw_root: Path, tasks: list[str], split: str):
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.raw_root, self.rows, self.cache = raw_root, [], {}
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
            path = structure_path(self.raw_root, task, mid, site)
            if not path.is_file():
                raise FileNotFoundError(f"Missing raw structure for {task} {mid}_{site:03d}: {path}")
            self.cache[key] = Structure.from_file(path) if path.name == "POSCAR" else parse_feff_structure(path)
        return task, self.cache[key], site, y


class CollateGraphs:
    def __init__(self, encoder):
        self.converter = Structure2Graph(encoder.element_types, encoder.cutoff)
        self.task_idx = {task: i for i, task in enumerate(FEFF_TASKS)}

    def graph(self, structure: Structure):
        out = self.converter.get_graph(structure)
        graph = out[0]
        lat = torch.tensor(np.asarray(structure.lattice.matrix, dtype=np.float32))
        if len(out) >= 2 and out[1] is not None:
            raw_lat = out[1]
            lat = torch.tensor(np.asarray(raw_lat[0] if getattr(raw_lat, "ndim", 0) == 3 else raw_lat, dtype=np.float32))
        if "pbc_offshift" in graph.edata:
            graph.edata["pbc_offshift"] = graph.edata["pbc_offshift"].to(torch.float32)
        else:
            graph.edata["pbc_offshift"] = graph.edata["pbc_offset"].to(torch.float32) @ lat
        if "pos" in graph.ndata:
            graph.ndata["pos"] = graph.ndata["pos"].to(torch.float32)
        elif "frac_coords" in graph.ndata:
            graph.ndata["pos"] = graph.ndata["frac_coords"].to(torch.float32) @ lat
        else:
            graph.ndata["pos"] = torch.tensor(np.asarray(structure.cart_coords, dtype=np.float32))
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


def load_feature_split(features: Path, task: str) -> MLSplits:
    return MLSplits(**{split: MLData(X=np.atleast_2d(np.loadtxt(features / f"{task}_{split}_X.txt", dtype=np.float32)), y=np.atleast_2d(np.loadtxt(features / f"{task}_{split}_y.txt", dtype=np.float32))) for split in SPLITS})


def missing_feature_splits(features: Path, tasks: list[str]) -> list[tuple[str, str]]:
    missing = []
    for task in tasks:
        for split in SPLITS:
            x_path, y_path = features / f"{task}_{split}_X.txt", features / f"{task}_{split}_y.txt"
            if not x_path.is_file() or not y_path.is_file():
                missing.append((task, split))
                continue
            X, y = np.atleast_2d(np.loadtxt(x_path, dtype=np.float32)), np.atleast_2d(np.loadtxt(y_path, dtype=np.float32))
            if X.shape[0] != y.shape[0] or X.shape[1] != 64 or y.shape[1] != 141 or not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"Invalid feature split {task} {split}: X={X.shape} y={y.shape}")
    return missing


class BalancedTaskBatchSampler(Sampler[list[int]]):
    """Yield exact, element-balanced batches and redraw large pools each epoch."""
    def __init__(self, rows: list[tuple[str, object, int, torch.Tensor]], rows_per_element: int, seed: int):
        if rows_per_element < 1:
            raise ValueError("rows_per_element must be positive")
        self.indices_by_task = {task: [] for task in FEFF_TASKS}
        for index, row in enumerate(rows):
            if row[0] not in self.indices_by_task:
                raise ValueError(f"Unexpected FEFF task in dataset: {row[0]}")
            self.indices_by_task[row[0]].append(index)
        counts = [len(self.indices_by_task[task]) for task in FEFF_TASKS]
        if any(count < rows_per_element for count in counts):
            raise ValueError(f"Each FEFF element needs at least {rows_per_element} rows: {dict(zip(FEFF_TASKS, counts))}")
        self.rows_per_element, self.batch_size, self.seed = int(rows_per_element), len(FEFF_TASKS) * int(rows_per_element), int(seed)
        self.epoch, self.batches_per_epoch = 0, min(counts) // rows_per_element
        if self.batches_per_epoch < 1:
            raise ValueError("Balanced sampler has no batches")

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        selected = {task: rng.permutation(self.indices_by_task[task])[: self.batches_per_epoch * self.rows_per_element] for task in FEFF_TASKS}
        for batch_number in range(self.batches_per_epoch):
            start, stop = batch_number * self.rows_per_element, (batch_number + 1) * self.rows_per_element
            batch = np.concatenate([selected[task][start:stop] for task in FEFF_TASKS])
            rng.shuffle(batch)
            yield batch.tolist()
