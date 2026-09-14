#!/usr/bin/env python3
"""Train the M3GNet XAS FEFF showcase pipeline.

Stages are: balanced scratch ``M3GNetXAS`` encoder training, 64D feature
export, balanced UniversalXAS training, and validation-selected tuned heads.

Prerequisites:
    From the repository root, run:
        bash tutorial_omnixas/download_omnixas_raw_data.sh
        export OMNIXAS_DATA_ROOT="$HOME/OmniXAS_data"

The download script downloads and extracts FEFF data by default. VASP download is
not needed for this FEFF pipeline. The shell script requires ``curl``,
``md5sum``, and ``tar``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

# DGL must receive its backend before any MatGL import loads DGL.
os.environ.setdefault("DGLBACKEND", "pytorch")

import random
import shutil
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from omnixas.model.m3gnet_xas import (
    FEATURE_DIM,
    FEATURE_SCALE,
    HEAD_HIDDEN_DIMS,
    SPECTRUM_DIM,
    M3GNetXAS,
    XASSpectralHead,
)

from omnixas.data.feff_graph import (
    BalancedTaskBatchSampler,
    CollateGraphs,
    ENCODER_BATCH,
    FEFFDataset,
    FEFF_TASKS,
    SPLITS,
    load_feature_split,
    missing_feature_splits,
    patch_matgl_gpu_constants,
    validate_raw_structures,
)

DEFAULT_EPOCHS = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default="output/training/m3gnet_xas_pipeline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preflight", "--self-check", action="store_true", help="Check inputs and architecture, then stop")
    p.add_argument("--evaluate", action="store_true", help="Evaluate completed checkpoints, without training")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--precision", choices=("32-true", "bf16-mixed"), default="32-true")
    p.add_argument("--encoder-epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--encoder-rows-per-element", type=int, default=12)
    p.add_argument("--encoder-eval-batch-size", type=int, default=ENCODER_BATCH)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--head-epochs", type=int, default=800, help="universal head max epochs")
    p.add_argument("--head-patience", type=int, default=60)
    p.add_argument("--tuned-epochs", type=int, default=1000, help="tuned head max epochs")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--prefetch-factor", type=int, default=2)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def preflight(root: Path, raw: Path) -> None:
    data = root / "tutorial_omnixas/ml_data"
    ids_dir = root / "tutorial_omnixas/material_id_and_site"
    counts: dict[str, int] = {}
    missing: list[str] = []
    material_sets: dict[str, set[str]] = {}
    for task in FEFF_TASKS:
        for split in SPLITS:
            x_path = data / f"{task}_{split}_X.txt"
            y_path = data / f"{task}_{split}_y.txt"
            id_path = ids_dir / f"{task}_{split}.txt"
            paths = (x_path, y_path, id_path)
            if not all(path.is_file() for path in paths):
                missing.extend(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing FEFF data files:\n" + "\n".join(missing[:12]))

    for task in FEFF_TASKS:
        for split in SPLITS:
            x_path = data / f"{task}_{split}_X.txt"
            y_path = data / f"{task}_{split}_y.txt"
            id_path = ids_dir / f"{task}_{split}.txt"
            X = np.atleast_2d(np.loadtxt(x_path, dtype=np.float32))
            y = np.atleast_2d(np.loadtxt(y_path, dtype=np.float32))
            rows = [line.strip() for line in id_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != len(set(rows)):
                raise ValueError(f"Duplicate full IDs in {id_path}")
            if X.shape != (len(rows), FEATURE_DIM):
                raise ValueError(f"Feature/ID order or shape mismatch for {task} {split}: X={X.shape}, IDs={len(rows)}")
            if y.shape != (len(rows), SPECTRUM_DIM):
                raise ValueError(f"Target/ID order or shape mismatch for {task} {split}: y={y.shape}, IDs={len(rows)}")
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite feature or target value for {task} {split}")
            material_sets[f"{task}:{split}"] = {row.rsplit("_", 1)[0] for row in rows}
            if split == "train": counts[task] = len(rows)
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = material_sets[f"{task}:{left}"] & material_sets[f"{task}:{right}"]
            if overlap:
                raise ValueError(f"Material leakage in {task}: {left}/{right}, example {next(iter(overlap))}")

    model = M3GNetXAS()
    params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in XASSpectralHead().parameters())
    print(json.dumps({"train_rows": sum(counts.values()), "feature_dim": FEATURE_DIM, "spectrum_dim": SPECTRUM_DIM, "encoder_parameters": params, "head_parameters": head_params, "raw_root": str(raw), "raw_root_exists": raw.is_dir()}, indent=2))
    if not raw.is_dir():
        raise FileNotFoundError(f"Missing raw FEFF structure root: {raw}. Set OMNIXAS_DATA_ROOT.")
    validate_raw_structures(root, raw, FEFF_TASKS)
    print("preflight passed: no training was started")


class LitScratch(pl.LightningModule):
    def __init__(self, model: M3GNetXAS, train_base: torch.Tensor, val_base: torch.Tensor, lr: float):
        super().__init__()
        self.model, self.lr = model, lr
        self.register_buffer("train_base", train_base)
        self.register_buffer("val_base", val_base)
        self.val_mse, self.val_task = [], []
        self.train_graph_cache_hits = 0
        self.train_graph_cache_builds = 0

    def step(self, batch, stage):
        graph = batch["graph"].to(self.device)
        line_graph = batch["line_graph"].to(self.device)
        site = batch["site"].to(self.device)
        pred = self.model(graph, line_graph, site)
        y = batch["y"].to(self.device)
        task = batch["task"].to(self.device)
        mse = ((pred - y) ** 2).mean(1)
        base = self.train_base if stage == "train" else self.val_base
        loss = (mse / base[task].clamp_min(1e-12)).mean() + 0.02 * (
            torch.diff(pred, dim=1) - torch.diff(y, dim=1)
        ).square().mean()
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        if stage == "val":
            self.val_mse.append(mse.detach())
            self.val_task.append(task.detach())
        return loss

    def on_train_epoch_start(self):
        self.train_graph_cache_hits = 0
        self.train_graph_cache_builds = 0

    def training_step(self, batch, _):
        self.train_graph_cache_hits += batch["graph_cache_hits"]
        self.train_graph_cache_builds += batch["graph_cache_builds"]
        return self.step(batch, "train")

    def on_train_epoch_end(self):
        total = self.train_graph_cache_hits + self.train_graph_cache_builds
        hit_rate = self.train_graph_cache_hits / total if total else 0.0
        self.log("train_graph_cache_hit_rate", hit_rate, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_graph_builds", self.train_graph_cache_builds, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):
        self.val_mse, self.val_task = [], []

    def validation_step(self, batch, _):
        return self.step(batch, "val")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.val_mse:
            return
        mses, tasks = torch.cat(self.val_mse), torch.cat(self.val_task)
        rel = [
            mses[tasks == i].median() / self.val_base[i].clamp_min(1e-12)
            for i in range(len(FEFF_TASKS))
            if (tasks == i).any()
        ]
        if len(rel) != len(FEFF_TASKS):
            raise RuntimeError("Validation lacks one or more FEFF elements")
        self.log("val_balanced_rel_mse", torch.stack(rel).mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
            fused=self.device.type == "cuda",
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=16, min_lr=1e-6)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_balanced_rel_mse"},
        }


def baselines(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    d = root / "tutorial_omnixas/ml_data"; train, val = [], []
    for task in FEFF_TASKS:
        ty = np.atleast_2d(np.loadtxt(d / f"{task}_train_y.txt", dtype=np.float32)); mean = ty.mean(0)
        train.append(np.median(((ty - mean) ** 2).mean(1)))
        vy = np.atleast_2d(np.loadtxt(d / f"{task}_val_y.txt", dtype=np.float32)); val.append(np.median(((vy - mean) ** 2).mean(1)))
    return torch.tensor(train), torch.tensor(val)


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f"Cannot write empty metrics file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows)))); w.writeheader(); w.writerows(rows)


def evaluate_head(head: XASSpectralHead, X: np.ndarray, y: np.ndarray, train_y: np.ndarray, batch: int) -> dict[str, float]:
    device = next(head.parameters()).device
    predictions = []
    with torch.inference_mode():
        for xb in torch.from_numpy(X).split(batch):
            predictions.append(head(xb.to(device, non_blocking=device.type == "cuda")).cpu().numpy())
    mse = np.mean((np.concatenate(predictions) - y) ** 2, axis=1)
    baseline = np.median(np.mean((y - train_y.mean(0)) ** 2, axis=1))
    return {
        "mse": float(mse.mean()),
        "median_mse": float(np.median(mse)),
        "baseline_median_mse": float(baseline),
        "eta": float(baseline / max(np.median(mse), 1e-12)),
    }

def validate_features(features: Path, root: Path) -> None:
    canonical = root / "tutorial_omnixas/ml_data"
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        for name in SPLITS:
            part = getattr(split, name)
            expected_y = np.atleast_2d(np.loadtxt(canonical / f"{task}_{name}_y.txt", dtype=np.float32))
            if part.X.shape != (len(part.y), FEATURE_DIM) or part.y.shape != (len(expected_y), SPECTRUM_DIM):
                raise ValueError(f"Invalid exported feature shape for {task} {name}")
            if not np.isfinite(part.X).all() or not np.isfinite(part.y).all():
                raise ValueError(f"Non-finite exported feature or target for {task} {name}")
            if not np.array_equal(part.y, expected_y):
                raise ValueError(f"Exported targets do not exactly match canonical targets: {task} {name}")


def balanced_universal(features: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splits = [load_feature_split(features, task) for task in FEFF_TASKS]
    n = max(len(s.train) for s in splits); rng = np.random.default_rng(seed)
    sampled = [rng.integers(len(s.train), size=n) for s in splits]
    train = [s.train.X[index] for s, index in zip(splits, sampled, strict=True)]
    y = [s.train.y[index] for s, index in zip(splits, sampled, strict=True)]
    return np.concatenate(train), np.concatenate(y), np.concatenate([s.val.X for s in splits]), np.concatenate([s.val.y for s in splits])


def train_head(out: Path, X: np.ndarray, y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray, source: dict | None, args: argparse.Namespace, *, lr: float, epochs: int, schedule: str, es_metric: str, label: str) -> Path:
    # settings ported from the balanced-activation pipeline best run (plain Adam, no weight decay)
    # patience=16 emulates the old pipeline's plateau (checked every 2 epochs with patience 8): identical LR-reduction timing (no improvement for 16 epochs)
    checkpoint = out / "best.pt"
    if out.exists() and not checkpoint.exists():
        raise RuntimeError(f"Head directory is incomplete and will not be reused: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        print(f"[{label}] reusing existing checkpoint: {checkpoint}", flush=True)
        return checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = XASSpectralHead()
    if source is not None:
        head.load_state_dict(source, strict=True)
    head.to(device)
    opt = torch.optim.Adam(
        head.parameters(),
        lr=lr,
        fused=device.type == "cuda",
    )
    best = float("inf")
    stale = 0
    best_epoch = -1
    if schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=16, min_lr=1e-6)
    elif schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown head schedule: {schedule}")
    if es_metric not in ("mean", "median"):
        raise ValueError(f"Unknown es_metric: {es_metric}")
    train_X_tensor = torch.as_tensor(X, device=device)
    train_y_tensor = torch.as_tensor(y, device=device)
    val_X_tensor = torch.as_tensor(val_X, device=device)
    val_y_tensor = torch.as_tensor(val_y, device=device)
    progress = tqdm(range(epochs), desc=label, unit="epoch")
    stopped_epoch = None
    for epoch in progress:
        head.train()
        indices = torch.randperm(len(train_X_tensor), device=device)
        for index in indices.split(args.batch_size):
            opt.zero_grad(set_to_none=True)
            loss = (head(train_X_tensor[index]) - train_y_tensor[index]).square().mean()
            loss.backward()
            opt.step()
        head.eval()
        with torch.inference_mode():
            val_mse = (head(val_X_tensor) - val_y_tensor).square().mean(1)
            val = float(val_mse.mean())
            val_metric = float(val_mse.median()) if es_metric == "median" else val
        if val_metric < best:
            best, stale, best_epoch = val_metric, 0, epoch
            torch.save({"state_dict": head.state_dict(), "epoch": epoch, "val_loss": val}, checkpoint)
        else:
            stale += 1
        if schedule == "plateau":
            sched.step(val)
        else:
            sched.step()
        progress.set_postfix(
            val=f"{val_metric:.2e}",
            best=f"{best:.2e}",
            lr=f"{opt.param_groups[0]['lr']:.2e}",
        )
        if stale >= args.head_patience:
            stopped_epoch = epoch + 1
            break
    progress.close()
    if stopped_epoch is not None:
        tqdm.write(f"[{label}] early stopping at epoch {stopped_epoch}/{epochs}, best at epoch {best_epoch}")
    if not checkpoint.is_file(): raise RuntimeError(f"Head training produced no checkpoint: {out}")
    tqdm.write(f"[{label}] done, checkpoint: {checkpoint}")
    return checkpoint


def load_state(path: Path) -> dict:
    if not path.is_file(): raise FileNotFoundError(f"Missing checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in state: raise ValueError(f"Checkpoint is missing state_dict: {path}")
    return state["state_dict"]


def write_evaluations(run: Path, features: Path, args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    universal_head = XASSpectralHead().to(device)
    universal_checkpoint = run / "heads/universalXAS/best.pt"
    universal_head.load_state_dict(load_state(universal_checkpoint), strict=True)
    universal_head.eval()
    universal_validation, universal_test = [], []
    tuned_validation, tuned_test = [], []
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        val = evaluate_head(universal_head, split.val.X, split.val.y, split.train.y, args.batch_size)
        test = evaluate_head(universal_head, split.test.X, split.test.y, split.train.y, args.batch_size)
        universal_validation.append({"dataset": task, "variant": "UniversalXAS", **{f"val_{k}": v for k, v in val.items()}})
        universal_test.append({"dataset": task, "variant": "UniversalXAS", **{f"test_{k}": v for k, v in test.items()}})

        checkpoint = run / f"heads/tunedUniversalXAS/{task}/best.pt"
        tuned_head = XASSpectralHead().to(device)
        tuned_head.load_state_dict(load_state(checkpoint), strict=True)
        tuned_head.eval()
        tuned_validation.append({"dataset": task, "checkpoint": str(checkpoint), **evaluate_head(tuned_head, split.val.X, split.val.y, split.train.y, args.batch_size)})
        tuned_test.append({"dataset": task, "checkpoint": str(checkpoint), **evaluate_head(tuned_head, split.test.X, split.test.y, split.train.y, args.batch_size)})
    csv_write(run / "universal_validation.csv", universal_validation)
    csv_write(run / "universal_test.csv", universal_test)
    csv_write(run / "tuned_validation.csv", tuned_validation)
    csv_write(run / "tuned_test.csv", tuned_test)


def ensure_file_limit(workers: int) -> None:
    """Check that the process can open enough file descriptors for DataLoader workers (Linux)."""
    if workers <= 0 or os.name != "posix":
        return
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        soft = hard
    need = 256 + 32 * workers
    proc_fd = Path("/proc/self/fd")
    open_fds = len(list(proc_fd.iterdir())) if proc_fd.is_dir() else 0
    if open_fds + need > soft:
        raise RuntimeError(
            f"Not enough open file descriptors for {workers} DataLoader workers: "
            f"{open_fds} open, limit {soft}, need about {need}. "
            "Restart the notebook kernel, raise the shell limit (ulimit -n 4096), or lower --num-workers."
        )


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.encoder_eval_batch_size < 1:
        raise ValueError("--encoder-eval-batch-size must be at least 1")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    ensure_file_limit(args.num_workers)
    if args.resume and args.overwrite: raise ValueError("Use either --resume or --overwrite, not both")
    if args.evaluate and args.overwrite: raise ValueError("--evaluate cannot be combined with --overwrite")
    if args.evaluate and args.resume: raise ValueError("--evaluate cannot be combined with --resume")
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    root = project_root()
    output = Path(args.output_root); output = output if output.is_absolute() else root / output
    if (args.resume or args.evaluate) and args.run_name is None:
        raise ValueError("--resume and --evaluate require --run-name")
    run = output / (args.run_name or f"m3gnet_xas_seed{args.seed}")
    if args.evaluate:
        if not (run / "RUN_COMPLETE.json").is_file():
            raise RuntimeError(f"Run is not complete. Missing {run / 'RUN_COMPLETE.json'}")
        load_state(run / "best_encoder.ckpt")
        features = run / "features"
        missing = [
            str(features / f"{task}_{split}_{suffix}.txt")
            for task in FEFF_TASKS
            for split in SPLITS
            for suffix in ("X", "y")
            if not (features / f"{task}_{split}_{suffix}.txt").is_file()
        ]
        if missing:
            raise FileNotFoundError("Completed run is missing feature artifacts:\n" + "\n".join(missing[:12]))
        validate_features(features, root)
        write_evaluations(run, features, args)
        print(f"evaluation complete: {run}")
        return
    if args.resume and not run.is_dir():
        raise FileNotFoundError(f"Cannot resume missing run directory: {run}")
    raw = Path(os.environ.get("OMNIXAS_DATA_ROOT", root.parent / "OmniXAS_data")) / "materialscloud_omnixas_raw" / "extracted"
    patch_matgl_gpu_constants(); preflight(root, raw)
    if args.preflight: return
    if args.overwrite and run.exists(): shutil.rmtree(run)
    if run.exists() and not args.resume: raise FileExistsError(f"Run exists: {run}; use --resume or --overwrite")
    if args.resume and (run / "RUN_COMPLETE.json").is_file():
        raise RuntimeError(f"Run is complete. Use --evaluate instead of --resume: {run}")
    run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); pl.seed_everything(args.seed, workers=True)
    if not args.resume:
        (run / "provenance.json").write_text(json.dumps({
            "pipeline": "m3gnet_xas_pipeline",
            "tasks": FEFF_TASKS,
            "architecture": {"model": "M3GNetXAS", "head": "XASSpectralHead", "feature_dim": FEATURE_DIM, "target_dim": SPECTRUM_DIM, "feature_scale": FEATURE_SCALE, "head_hidden_dims": list(HEAD_HIDDEN_DIMS)},
            "balancing": {"policy": "sample each FEFF train split with replacement to the largest task count", "seed": args.seed},
            "selection": "validation loss/eta only. Test metrics are reported after selection.",
            "args": vars(args),
        }, indent=2), encoding="utf-8")
    train_base, val_base = baselines(root)
    graph_loader_kwargs = {"num_workers": args.num_workers}
    if args.num_workers > 0:
        graph_loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    encoder_path = run / "best_encoder.ckpt"
    if not encoder_path.exists():
        model = M3GNetXAS(); collate = CollateGraphs(model.encoder); train_ds = FEFFDataset(root, raw, FEFF_TASKS, "train")
        task_counts = {task: sum(row[0] == task for row in train_ds.rows) for task in FEFF_TASKS}
        rows_per_element = min(args.encoder_rows_per_element, min(task_counts.values()))
        sampler = BalancedTaskBatchSampler(train_ds.rows, rows_per_element, args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, **graph_loader_kwargs)
        val_loader = DataLoader(
            FEFFDataset(root, raw, FEFF_TASKS, "val"),
            batch_size=args.encoder_eval_batch_size,
            collate_fn=collate,
            **graph_loader_kwargs,
        )
        print(
            f"Encoder batch sizes: train={sampler.batch_size} "
            f"({rows_per_element} rows per element), eval={args.encoder_eval_batch_size}",
            flush=True,
        )
        cb = ModelCheckpoint(run / "encoder_checkpoints", filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}", monitor="val_balanced_rel_mse", mode="min", save_top_k=1, save_last=True)
        trainer = pl.Trainer(max_epochs=args.encoder_epochs, accelerator="auto", devices=1, precision=args.precision, callbacks=[cb, EarlyStopping(monitor="val_balanced_rel_mse", patience=60, mode="min")], logger=CSVLogger(str(run), name="encoder_logs"), log_every_n_steps=10)
        trainer.fit(LitScratch(model, train_base, val_base, args.encoder_lr), train_loader, val_loader, ckpt_path=str(run / "encoder_checkpoints/last.ckpt") if args.resume and (run / "encoder_checkpoints/last.ckpt").exists() else None)
        if not cb.best_model_path: raise RuntimeError("Encoder training produced no validation checkpoint")
        shutil.copy2(cb.best_model_path, encoder_path)
    features = run / "features"; features.mkdir(exist_ok=True); missing = missing_feature_splits(features, FEFF_TASKS)
    expected = [(task, split) for task in FEFF_TASKS for split in SPLITS]
    if missing and len(missing) != len(expected):
        raise RuntimeError("Feature directory is incomplete. Remove it only with --overwrite, then regenerate all features.")
    model = M3GNetXAS(); state = torch.load(encoder_path, map_location="cpu", weights_only=False)["state_dict"]; model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True); model.eval(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); collate = CollateGraphs(model.encoder)
    with torch.inference_mode():
        for task, split in tqdm(missing, desc="M3GNet feature export", unit="split", disable=not missing):
            loader = DataLoader(
                FEFFDataset(root, raw, [task], split),
                batch_size=args.encoder_eval_batch_size,
                collate_fn=collate,
                **graph_loader_kwargs,
            )
            xs, ys = [], []
            for b in loader:
                xs.append(model.encode(b["graph"].to(device), b["line_graph"].to(device), b["site"].to(device), scaled=True).cpu().numpy())
                ys.append(b["y"].numpy())
            X, y = np.concatenate(xs), np.concatenate(ys); np.savetxt(features / f"{task}_{split}_X.txt", X); np.savetxt(features / f"{task}_{split}_y.txt", y)
    validate_features(features, root)
    ux, uy, uvx, uvy = balanced_universal(features, args.seed)
    universal = train_head(run / "heads/universalXAS", ux, uy, uvx, uvy, None, args, lr=5e-4, epochs=args.head_epochs, schedule="plateau", es_metric="mean", label="UniversalXAS")
    universal_state = load_state(universal)
    for i, task in enumerate(FEFF_TASKS):
        split = load_feature_split(features, task)
        train_head(run / f"heads/tunedUniversalXAS/{task}", split.train.X, split.train.y, split.val.X, split.val.y, universal_state, args, lr=3e-4, epochs=args.tuned_epochs, schedule="cosine", es_metric="median", label=f"Tuned {task} ({i + 1}/{len(FEFF_TASKS)})")
    print("Evaluating heads on val and test splits...", flush=True)
    write_evaluations(run, features, args)
    (run / "RUN_COMPLETE.json").write_text(json.dumps({"status": "complete", "selection": "validation metrics only, test evaluated once", "encoder": str(encoder_path), "universal": str(universal)}, indent=2), encoding="utf-8")
    print(f"Run complete: {run}", flush=True)


if __name__ == "__main__": main()
