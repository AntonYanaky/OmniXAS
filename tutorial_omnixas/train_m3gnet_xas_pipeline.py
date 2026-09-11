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
import random
import shutil
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from omnixas.model.m3gnet_xas import (
    FEATURE_DIM,
    FEATURE_SCALE,
    HEAD_HIDDEN_DIMS,
    SPECTRUM_DIM,
    M3GNetXAS,
    XASSpectralHead,
)

from train_all8_feff import (  # existing FEFF graph and row-alignment utilities
    CollateGraphs, FEFFDataset, FEFF_TASKS, ENCODER_BATCH, patch_matgl_gpu_constants,
    validate_raw_structures, load_feature_split, missing_feature_splits,
)
from train_e2e_balanced_feff import BalancedTaskBatchSampler

SPLITS = ("train", "val", "test")
DEFAULT_EPOCHS = 300


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
    p.add_argument("--encoder-epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--head-epochs", type=int, default=800)
    p.add_argument("--head-patience", type=int, default=60)
    p.add_argument("--head-lr", type=float, default=7e-4)
    p.add_argument("--batch-size", type=int, default=96)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root(root: Path) -> Path:
    return Path(os.environ.get("OMNIXAS_DATA_ROOT", root.parent / "OmniXAS_data")) / "materialscloud_omnixas_raw" / "extracted"


def _ids(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate full IDs in {path}")
    return rows


def check_arrays(root: Path) -> dict[str, int]:
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
            rows = _ids(id_path)
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
    return counts


def preflight(root: Path, raw: Path) -> None:
    counts = check_arrays(root)
    model = M3GNetXAS()
    params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in XASSpectralHead().parameters())
    print(json.dumps({"tasks": FEFF_TASKS, "train_rows": counts, "feature_dim": FEATURE_DIM, "spectrum_dim": SPECTRUM_DIM, "encoder_parameters": params, "head_parameters": head_params, "raw_root": str(raw), "raw_root_exists": raw.is_dir()}, indent=2))
    if not raw.is_dir():
        raise FileNotFoundError(f"Missing raw FEFF structure root: {raw}. Set OMNIXAS_DATA_ROOT.")
    validate_raw_structures(root, raw, FEFF_TASKS)
    print("preflight passed: no training was started")


class LitScratch(pl.LightningModule):
    def __init__(self, model: M3GNetXAS, train_base: torch.Tensor, val_base: torch.Tensor, lr: float, epochs: int):
        super().__init__(); self.model, self.lr, self.epochs = model, lr, epochs
        self.register_buffer("train_base", train_base); self.register_buffer("val_base", val_base)
        self.val_mse, self.val_task = [], []

    def step(self, batch, stage):
        pred = self.model(batch["graph"].to(self.device), batch["site"].to(self.device)); y = batch["y"].to(self.device); task = batch["task"].to(self.device)
        mse = ((pred - y) ** 2).mean(1); base = self.train_base if stage == "train" else self.val_base
        loss = (mse / base[task].clamp_min(1e-12)).mean() + 0.02 * (torch.diff(pred, dim=1) - torch.diff(y, dim=1)).square().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        if stage == "val": self.val_mse.append(mse.detach()); self.val_task.append(task.detach())
        return loss

    def training_step(self, batch, _): return self.step(batch, "train")
    def on_validation_epoch_start(self): self.val_mse, self.val_task = [], []
    def validation_step(self, batch, _): return self.step(batch, "val")
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.val_mse: return
        mses, tasks = torch.cat(self.val_mse), torch.cat(self.val_task)
        rel = [mses[tasks == i].median() / self.val_base[i].clamp_min(1e-12) for i in range(len(FEFF_TASKS)) if (tasks == i).any()]
        if len(rel) != len(FEFF_TASKS): raise RuntimeError("Validation lacks one or more FEFF elements")
        self.log("val_balanced_rel_mse", torch.stack(rel).mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs, eta_min=1e-6), "interval": "epoch"}}


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


def predict_head(state: dict, X: np.ndarray, y: np.ndarray, train_y: np.ndarray, batch: int) -> dict[str, float]:
    head = XASSpectralHead(); head.load_state_dict(state, strict=True); head.eval(); pred = []
    with torch.inference_mode():
        for (xb,) in DataLoader(TensorDataset(torch.tensor(X)), batch_size=batch): pred.append(head(xb).numpy())
    pred = np.concatenate(pred); mse = np.mean((pred - y) ** 2, 1); base = np.median(np.mean((y - train_y.mean(0)) ** 2, 1))
    return {"mse": float(mse.mean()), "median_mse": float(np.median(mse)), "baseline_median_mse": float(base), "eta": float(base / max(np.median(mse), 1e-12))}


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


def train_head(out: Path, X: np.ndarray, y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray, source: dict | None, args: argparse.Namespace) -> Path:
    checkpoint = out / "best.pt"
    if out.exists() and not checkpoint.exists():
        raise RuntimeError(f"Head directory is incomplete and will not be reused: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists(): return checkpoint
    head = XASSpectralHead();
    if source is not None: head.load_state_dict(source, strict=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-5); best = float("inf"); stale = 0
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.head_epochs):
        head.train()
        for xb, yb in loader: opt.zero_grad(); loss = (head(xb) - yb).square().mean(); loss.backward(); opt.step()
        head.eval()
        with torch.inference_mode(): val = float((head(torch.tensor(val_X)) - torch.tensor(val_y)).square().mean())
        if val < best: best, stale = val, 0; torch.save({"state_dict": head.state_dict(), "epoch": epoch, "val_loss": val}, checkpoint)
        else: stale += 1
        if stale >= args.head_patience: break
    if not checkpoint.is_file(): raise RuntimeError(f"Head training produced no checkpoint: {out}")
    return checkpoint


def load_state(path: Path) -> dict:
    if not path.is_file(): raise FileNotFoundError(f"Missing checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in state: raise ValueError(f"Checkpoint is missing state_dict: {path}")
    return state["state_dict"]


def evaluate_run(run: Path, args: argparse.Namespace) -> None:
    if not (run / "RUN_COMPLETE.json").is_file():
        raise RuntimeError(f"Run is not complete. Missing {run / 'RUN_COMPLETE.json'}")
    encoder = run / "best_encoder.ckpt"
    load_state(encoder)
    features = run / "features"
    required = [features / f"{task}_{split}_{suffix}.txt" for task in FEFF_TASKS for split in SPLITS for suffix in ("X", "y")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Completed run is missing feature artifacts:\n" + "\n".join(missing[:12]))
    validate_features(features, project_root())
    universal = run / "heads/universalXAS/best.pt"
    universal_state = load_state(universal)
    validation_rows, test_rows = [], []
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        val = predict_head(universal_state, split.val.X, split.val.y, split.train.y, args.batch_size)
        test = predict_head(universal_state, split.test.X, split.test.y, split.train.y, args.batch_size)
        validation_rows.append({"dataset": task, "variant": "UniversalXAS", **{f"val_{k}": v for k, v in val.items()}})
        test_rows.append({"dataset": task, "variant": "UniversalXAS", **{f"test_{k}": v for k, v in test.items()}})
    csv_write(run / "universal_validation.csv", validation_rows)
    csv_write(run / "universal_test.csv", test_rows)
    tuned_rows, tuned_test_rows = [], []
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        checkpoint = run / f"heads/tunedUniversalXAS/{task}/best.pt"
        state = load_state(checkpoint)
        tuned_rows.append({"dataset": task, "checkpoint": str(checkpoint), **predict_head(state, split.val.X, split.val.y, split.train.y, args.batch_size)})
        tuned_test_rows.append({"dataset": task, "checkpoint": str(checkpoint), **predict_head(state, split.test.X, split.test.y, split.train.y, args.batch_size)})
    csv_write(run / "tuned_validation.csv", tuned_rows)
    csv_write(run / "tuned_test.csv", tuned_test_rows)
    print(f"evaluation complete: {run}")


def main() -> None:
    args = parse_args()
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
        evaluate_run(run, args)
        return
    if args.resume and not run.is_dir():
        raise FileNotFoundError(f"Cannot resume missing run directory: {run}")
    raw = data_root(root)
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
    encoder_path = run / "best_encoder.ckpt"
    if not encoder_path.exists():
        model = M3GNetXAS(); collate = CollateGraphs(model.encoder); train_ds = FEFFDataset(root, raw, FEFF_TASKS, "train")
        task_counts = {task: sum(row[0] == task for row in train_ds.rows) for task in FEFF_TASKS}
        rows_per_element = min(12, min(task_counts.values()))
        sampler = BalancedTaskBatchSampler(train_ds.rows, rows_per_element, args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.num_workers)
        val_loader = DataLoader(FEFFDataset(root, raw, FEFF_TASKS, "val"), batch_size=ENCODER_BATCH, collate_fn=collate, num_workers=args.num_workers)
        cb = ModelCheckpoint(run / "encoder_checkpoints", filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}", monitor="val_balanced_rel_mse", mode="min", save_top_k=1, save_last=True)
        trainer = pl.Trainer(max_epochs=args.encoder_epochs, accelerator="auto", devices=1, callbacks=[cb, EarlyStopping(monitor="val_balanced_rel_mse", patience=30, mode="min")], logger=CSVLogger(str(run), name="encoder_logs"), log_every_n_steps=1)
        trainer.fit(LitScratch(model, train_base, val_base, args.encoder_lr, args.encoder_epochs), train_loader, val_loader, ckpt_path=str(run / "encoder_checkpoints/last.ckpt") if args.resume and (run / "encoder_checkpoints/last.ckpt").exists() else None)
        if not cb.best_model_path: raise RuntimeError("Encoder training produced no validation checkpoint")
        shutil.copy2(cb.best_model_path, encoder_path)
    features = run / "features"; features.mkdir(exist_ok=True); missing = missing_feature_splits(features, FEFF_TASKS)
    expected = [(task, split) for task in FEFF_TASKS for split in SPLITS]
    if missing and len(missing) != len(expected):
        raise RuntimeError("Feature directory is incomplete. Remove it only with --overwrite, then regenerate all features.")
    model = M3GNetXAS(); state = torch.load(encoder_path, map_location="cpu", weights_only=False)["state_dict"]; model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True); model.eval(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); collate = CollateGraphs(model.encoder)
    with torch.inference_mode():
        for task, split in missing:
            loader = DataLoader(FEFFDataset(root, raw, [task], split), batch_size=ENCODER_BATCH, collate_fn=collate, num_workers=args.num_workers); xs, ys = [], []
            for b in loader:
                xs.append(model.encode(b["graph"].to(device), b["site"].to(device), scaled=True).cpu().numpy())
                ys.append(b["y"].numpy())
            X, y = np.concatenate(xs), np.concatenate(ys); np.savetxt(features / f"{task}_{split}_X.txt", X); np.savetxt(features / f"{task}_{split}_y.txt", y)
    validate_features(features, root)
    ux, uy, uvx, uvy = balanced_universal(features, args.seed); universal = train_head(run / "heads/universalXAS", ux, uy, uvx, uvy, None, args); universal_state = load_state(universal)
    validation_rows, test_rows = [], []
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        val = predict_head(universal_state, split.val.X, split.val.y, split.train.y, args.batch_size)
        test = predict_head(universal_state, split.test.X, split.test.y, split.train.y, args.batch_size)
        validation_rows.append({"dataset": task, "variant": "UniversalXAS", **{f"val_{k}": v for k, v in val.items()}})
        test_rows.append({"dataset": task, "variant": "UniversalXAS", **{f"test_{k}": v for k, v in test.items()}})
    csv_write(run / "universal_validation.csv", validation_rows)
    csv_write(run / "universal_test.csv", test_rows)
    tuned_rows, tuned_test_rows = [], []
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        ckpt = train_head(run / f"heads/tunedUniversalXAS/{task}", split.train.X, split.train.y, split.val.X, split.val.y, universal_state, args)
        val = predict_head(load_state(ckpt), split.val.X, split.val.y, split.train.y, args.batch_size)
        test = predict_head(load_state(ckpt), split.test.X, split.test.y, split.train.y, args.batch_size)
        tuned_rows.append({"dataset": task, "checkpoint": str(ckpt), **val})
        tuned_test_rows.append({"dataset": task, "checkpoint": str(ckpt), **test})
    csv_write(run / "tuned_validation.csv", tuned_rows)
    csv_write(run / "tuned_test.csv", tuned_test_rows)
    (run / "RUN_COMPLETE.json").write_text(json.dumps({"status": "complete", "selection": "validation metrics only, test evaluated once", "encoder": str(encoder_path), "universal": str(universal)}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
