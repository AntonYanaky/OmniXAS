#!/usr/bin/env python3
"""Train a balanced FEFF-only end-to-end encoder and XAS heads.

The pipeline trains a scratch encoder with an element-balanced graph loader,
exports frozen FEFF features, trains one UniversalXAS head, and fine-tunes one
Tuned-UniversalXAS head for each FEFF element.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Sampler, TensorDataset

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.metrics import ModelMetrics
from omnixas.model.training import PlModule
from omnixas.model.xasblock import XASBlock

from train_all8_feff import (
    ENCODER_BATCH,
    ENCODER_DERIV_LAMBDA,
    FEFF_TASKS,
    FEATURE_SCALE,
    HEAD_WIDTHS,
    INPUT_DIM,
    OUTPUT_DIM,
    CollateGraphs,
    FEFFDataset,
    checkpoint_score,
    csv_write,
    load_feature_split,
    missing_feature_splits,
    patch_matgl_gpu_constants,
    torch_load,
    validate_raw_structures,
)
from train_e2e_custom_universal import (
    EndToEndUniversal,
    best_checkpoint,
    build_head_regressor,
    head_state_from_e2e,
)


E2E_MIN_LR = 1e-6
HEAD_BATCH_SIZE = 48
UNIVERSAL_SEED = 44
UNIVERSAL_LR = 5e-4
UNIVERSAL_DROPOUT = 0.10
UNIVERSAL_EPOCHS = 1000
UNIVERSAL_PATIENCE = 60
UNIVERSAL_PLATEAU_FACTOR = 0.5
UNIVERSAL_PLATEAU_PATIENCE = 8
UNIVERSAL_VARIANT = "feff_plateau_lr5e-4_do010_seed44"
TUNED_SETTING = {
    "name": "cosT500_lr3e-4_do0p15_p60",
    "seed": 145,
    "dropout": 0.15,
    "lr": 3e-4,
    "cosine_t": 500,
    "eta_min": 1e-6,
    "batch_size": HEAD_BATCH_SIZE,
    "epochs": 1000,
    "patience": 60,
}


class BalancedTaskBatchSampler(Sampler[list[int]]):
    """Yield exact, element-balanced batches and redraw large pools each epoch."""

    def __init__(
        self,
        rows: list[tuple[str, Any, int, torch.Tensor]],
        rows_per_element: int,
        seed: int,
    ):
        if rows_per_element < 1:
            raise ValueError("rows_per_element must be positive")
        self.indices_by_task: dict[str, list[int]] = {task: [] for task in FEFF_TASKS}
        for index, row in enumerate(rows):
            task = row[0]
            if task not in self.indices_by_task:
                raise ValueError(f"Unexpected FEFF task in dataset: {task}")
            self.indices_by_task[task].append(index)
        counts = [len(self.indices_by_task[task]) for task in FEFF_TASKS]
        if any(count < rows_per_element for count in counts):
            counts_by_task = dict(zip(FEFF_TASKS, counts))
            raise ValueError(
                f"Each FEFF element needs at least {rows_per_element} rows: {counts_by_task}"
            )
        self.rows_per_element = int(rows_per_element)
        self.batch_size = len(FEFF_TASKS) * self.rows_per_element
        self.seed = int(seed)
        self.epoch = 0
        self.batches_per_epoch = min(counts) // self.rows_per_element
        if self.batches_per_epoch < 1:
            raise ValueError("Balanced sampler has no batches")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        rows_to_select = self.batches_per_epoch * self.rows_per_element
        selected = {
            task: rng.permutation(self.indices_by_task[task])[:rows_to_select]
            for task in FEFF_TASKS
        }
        for batch_number in range(self.batches_per_epoch):
            start = batch_number * self.rows_per_element
            stop = (batch_number + 1) * self.rows_per_element
            batch = np.concatenate([selected[task][start:stop] for task in FEFF_TASKS])
            rng.shuffle(batch)
            yield batch.tolist()


class LitBalancedEndToEnd(pl.LightningModule):
    def __init__(
        self,
        model: EndToEndUniversal,
        tasks: list[str],
        train_base: torch.Tensor,
        val_base: torch.Tensor,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
        epochs: int,
        scheduler: str,
        plateau_factor: float,
        plateau_patience: int,
    ):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.scheduler = scheduler
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.register_buffer("train_base", train_base.float())
        self.register_buffer("val_base", val_base.float())
        self.val_mses: list[torch.Tensor] = []
        self.val_tasks: list[torch.Tensor] = []

    def step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        graph = batch["graph"].to(self.device)
        site = batch["site"].to(self.device)
        task = batch["task"].to(self.device)
        y = batch["y"].to(self.device)
        pred = self.model(graph, site)
        sample_mse = ((pred - y) ** 2).mean(dim=1)
        baseline = self.train_base if stage == "train" else self.val_base
        loss = (sample_mse / baseline[task].clamp_min(1e-12)).mean()
        derivative_mse = (torch.diff(pred, dim=1) - torch.diff(y, dim=1)) ** 2
        loss = loss + ENCODER_DERIV_LAMBDA * derivative_mse.mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mse", sample_mse.mean(), on_epoch=True, prog_bar=True)
        if stage == "val":
            self.val_mses.append(sample_mse.detach())
            self.val_tasks.append(task.detach())
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self.step(batch, "train")

    def on_validation_epoch_start(self) -> None:
        self.val_mses = []
        self.val_tasks = []

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self.step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if not self.val_mses:
            raise RuntimeError("Validation produced no rows")
        mses = torch.cat(self.val_mses)
        tasks = torch.cat(self.val_tasks)
        relative_medians = []
        for index, task_name in enumerate(self.tasks):
            mask = tasks == index
            if not mask.any():
                raise RuntimeError(f"Validation has no rows for {task_name}")
            median_mse = mses[mask].median()
            relative_medians.append(median_mse / self.val_base[index].clamp_min(1e-12))
            self.log(
                f"val_eta_{task_name}",
                self.val_base[index] / median_mse.clamp_min(1e-12),
                on_epoch=True,
                prog_bar=(task_name == "Cu_FEFF"),
            )
        self.log(
            "val_balanced_rel_mse",
            torch.stack(relative_medians).mean(),
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.model.head.parameters(), "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )
        if self.scheduler == "none":
            return optimizer
        if self.scheduler == "cosine":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.epochs, eta_min=E2E_MIN_LR
                    ),
                    "interval": "epoch",
                },
            }
        if self.scheduler == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=self.plateau_factor,
                        patience=self.plateau_patience,
                        min_lr=E2E_MIN_LR,
                    ),
                    "interval": "epoch",
                    "frequency": 2,
                    "monitor": "val_balanced_rel_mse",
                },
            }
        raise ValueError(f"Unsupported E2E scheduler: {self.scheduler}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="output/training/balancedE2EFeff")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gnn-dropout", type=float, default=0.10)
    parser.add_argument("--encoder-lr", type=float, default=1e-3)
    parser.add_argument("--e2e-head-lr", type=float, default=5e-4)
    parser.add_argument("--e2e-head-dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--e2e-scheduler",
        choices=["plateau", "cosine", "none"],
        default="plateau",
    )
    parser.add_argument("--e2e-plateau-factor", type=float, default=0.5)
    parser.add_argument("--e2e-plateau-patience", type=int, default=8)
    parser.add_argument("--e2e-epochs", type=int, default=1000)
    parser.add_argument("--e2e-patience", type=int, default=60)
    parser.add_argument("--rows-per-element", type=int, default=6)
    return parser.parse_args()


def task_baselines(
    root: Path,
    tasks: list[str],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    data_dir = root / "tutorial_omnixas" / "ml_data"
    train_base, val_base, counts = [], [], {}
    for task in tasks:
        train_y = np.atleast_2d(np.loadtxt(data_dir / f"{task}_train_y.txt", dtype=np.float32))
        counts[task] = len(train_y)
        if train_y.shape[1] != OUTPUT_DIM:
            raise ValueError(f"Invalid target shape for {task} train: {train_y.shape}")
        val_y = np.atleast_2d(np.loadtxt(data_dir / f"{task}_val_y.txt", dtype=np.float32))
        if val_y.shape[1] != OUTPUT_DIM:
            raise ValueError(f"Invalid target shape for {task} val: {val_y.shape}")
        for y, output in ((train_y, train_base), (val_y, val_base)):
            baseline = np.repeat(train_y.mean(axis=0, keepdims=True), len(y), axis=0)
            output.append(float(np.median(np.mean((y - baseline) ** 2, axis=1))))
    return torch.tensor(train_base), torch.tensor(val_base), counts


def ensure_clean_or_complete(directory: Path, label: str) -> None:
    checkpoint = best_checkpoint(directory)
    if checkpoint is None and directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Incomplete {label} exists without a best checkpoint: {directory}")


def load_e2e_model(
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> EndToEndUniversal:
    model = EndToEndUniversal(args.gnn_dropout, args.e2e_head_dropout)
    state = torch_load(checkpoint).get("state_dict")
    if state is None:
        raise ValueError(f"Checkpoint is missing state_dict: {checkpoint}")
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)
    return model.to(device).eval()


def train_e2e(
    run: Path,
    root: Path,
    raw_root: Path,
    args: argparse.Namespace,
    train_base: torch.Tensor,
    val_base: torch.Tensor,
    train_dataset: FEFFDataset,
    effective_plateau_patience: int,
    effective_early_stopping_patience: int,
) -> Path:
    copied = run / "best_e2e_encoder_universal.ckpt"
    if copied.is_file():
        print(f"reusing end-to-end checkpoint: {copied}", flush=True)
        return copied
    ensure_clean_or_complete(run / "checkpoints", "E2E checkpoints")
    model = EndToEndUniversal(args.gnn_dropout, args.e2e_head_dropout)
    collate = CollateGraphs(model.encoder)
    sampler = BalancedTaskBatchSampler(train_dataset.rows, args.rows_per_element, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FEFFDataset(root, raw_root, FEFF_TASKS, "val"),
        batch_size=ENCODER_BATCH,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    module = LitBalancedEndToEnd(
        model,
        FEFF_TASKS,
        train_base,
        val_base,
        args.encoder_lr,
        args.e2e_head_lr,
        args.weight_decay,
        args.e2e_epochs,
        args.e2e_scheduler,
        args.e2e_plateau_factor,
        effective_plateau_patience,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run / "checkpoints",
        filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}",
        monitor="val_balanced_rel_mse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.e2e_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_balanced_rel_mse",
                patience=effective_early_stopping_patience,
                mode="min",
            ),
            checkpoint,
        ],
        logger=CSVLogger(save_dir=str(run), name="e2e_logs"),
        log_every_n_steps=1,
    )
    trainer.fit(module, train_loader, val_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("E2E training finished without a best checkpoint")
    shutil.copy2(checkpoint.best_model_path, copied)
    return copied


def export_features(
    run: Path,
    checkpoint: Path,
    root: Path,
    raw_root: Path,
    args: argparse.Namespace,
) -> Path:
    features = run / "features"
    missing = missing_feature_splits(features, FEFF_TASKS)
    if not missing:
        print("all FEFF features already present", flush=True)
        return features
    if features.exists() and any(features.iterdir()):
        raise RuntimeError(f"Incomplete non-empty feature directory: {features}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_e2e_model(checkpoint, args, device)
    collate = CollateGraphs(model.encoder)
    features.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for task, split_name in missing:
            loader = DataLoader(
                FEFFDataset(root, raw_root, [task], split_name),
                batch_size=ENCODER_BATCH,
                shuffle=False,
                collate_fn=collate,
                num_workers=args.num_workers,
            )
            outputs, targets = [], []
            for batch in loader:
                z = model.encoder(batch["graph"].to(device), batch["site"].to(device))
                outputs.append((z * FEATURE_SCALE).cpu().numpy())
                targets.append(batch["y"].numpy())
            X = np.concatenate(outputs)
            y = np.concatenate(targets)
            if X.shape != (len(y), INPUT_DIM) or y.shape[1] != OUTPUT_DIM:
                raise ValueError(f"Invalid export for {task} {split_name}: X={X.shape} y={y.shape}")
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite export for {task} {split_name}")
            np.savetxt(features / f"{task}_{split_name}_X.txt", X)
            np.savetxt(features / f"{task}_{split_name}_y.txt", y)
            print(f"exported {task} {split_name}: X={X.shape} y={y.shape}", flush=True)
    if missing_feature_splits(features, FEFF_TASKS):
        raise RuntimeError("FEFF feature export did not complete")
    return features


def universal_split(features: Path) -> MLSplits:
    splits = [load_feature_split(features, task) for task in FEFF_TASKS]
    return MLSplits(
        train=MLData(
            X=np.concatenate([split.train.X for split in splits]),
            y=np.concatenate([split.train.y for split in splits]),
        ),
        val=MLData(
            X=np.concatenate([split.val.X for split in splits]),
            y=np.concatenate([split.val.y for split in splits]),
        ),
        test=MLData(
            X=np.concatenate([split.test.X for split in splits]),
            y=np.concatenate([split.test.y for split in splits]),
        ),
    )


def evaluate_e2e(
    checkpoint: Path,
    root: Path,
    raw_root: Path,
    run: Path,
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_e2e_model(checkpoint, args, device)
    collate = CollateGraphs(model.encoder)
    rows = []
    train_base, _, _ = task_baselines(root, FEFF_TASKS)
    for index, task in enumerate(FEFF_TASKS):
        dataset = FEFFDataset(root, raw_root, [task], "val")
        loader = DataLoader(
            dataset,
            batch_size=ENCODER_BATCH,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.num_workers,
        )
        predictions, targets = [], []
        with torch.inference_mode():
            for batch in loader:
                predictions.append(
                    model(
                        batch["graph"].to(device),
                        batch["site"].to(device),
                    ).cpu().numpy()
                )
                targets.append(batch["y"].numpy())
        pred = np.concatenate(predictions)
        y = np.concatenate(targets)
        median_mse = float(np.median(np.mean((pred - y) ** 2, axis=1)))
        baseline = float(train_base[index])
        rows.append({
            "dataset": task,
            "split": "val",
            "checkpoint": str(checkpoint),
            "median_mse": median_mse,
            "baseline_median_mse": baseline,
            "eta": baseline / median_mse,
        })
    csv_write(run / "e2e_eval.csv", rows)


def train_universal(features: Path, e2e_checkpoint: Path, run: Path) -> Path:
    out = run / "heads" / "universalXAS" / UNIVERSAL_VARIANT
    checkpoint = best_checkpoint(out)
    if checkpoint is not None:
        print(f"reusing UniversalXAS checkpoint: {checkpoint}", flush=True)
        return checkpoint
    ensure_clean_or_complete(out, "UniversalXAS")
    pl.seed_everything(UNIVERSAL_SEED, workers=True)
    universal = build_head_regressor(
        out,
        lr=UNIVERSAL_LR,
        dropout=UNIVERSAL_DROPOUT,
        batch=HEAD_BATCH_SIZE,
        epochs=UNIVERSAL_EPOCHS,
        patience=UNIVERSAL_PATIENCE,
        scheduler="plateau",
        cosine_t=UNIVERSAL_EPOCHS,
        plateau_factor=UNIVERSAL_PLATEAU_FACTOR,
        plateau_patience=UNIVERSAL_PLATEAU_PATIENCE,
    )
    universal.model.model.load_state_dict(head_state_from_e2e(e2e_checkpoint), strict=True)
    universal.fit(universal_split(features))
    return Path(universal.cfg.fetch_checkpoint("best"))


def evaluate_head(checkpoint: Path, split: MLSplits, split_name: str) -> dict[str, float]:
    X = getattr(split, split_name).X
    y = getattr(split, split_name).y
    module = PlModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        model=XASBlock(INPUT_DIM, HEAD_WIDTHS, OUTPUT_DIM),
        lr=1e-4,
    ).eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    predictions = []
    with torch.inference_mode():
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        for (xb,) in DataLoader(dataset, batch_size=1024):
            predictions.append(module(xb.to(device)).cpu().numpy())
    pred = np.concatenate(predictions)
    median_mse = float(ModelMetrics(predictions=pred, targets=y).median_of_mse_per_spectra)
    baseline = np.repeat(split.train.y.mean(axis=0, keepdims=True), len(y), axis=0)
    baseline_mse = float(np.median(np.mean((y - baseline) ** 2, axis=1)))
    return {
        f"{split_name}_median_mse": median_mse,
        f"{split_name}_baseline_median_mse": baseline_mse,
        f"{split_name}_eta": baseline_mse / median_mse,
    }


def evaluate_universal(checkpoint: Path, features: Path, run: Path) -> None:
    rows = []
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        row = {
            "dataset": task,
            "variant": UNIVERSAL_VARIANT,
            "checkpoint": str(checkpoint),
            "val_loss_score": checkpoint_score(checkpoint),
        }
        row.update(evaluate_head(checkpoint, split, "val"))
        row.update(evaluate_head(checkpoint, split, "test"))
        rows.append(row)
        print(
            f"universal {task}: val_eta={row['val_eta']:.4f} "
            f"test_eta={row['test_eta']:.4f}",
            flush=True,
        )
    csv_write(run / "universal_eval.csv", rows)


def train_tuned(checkpoint: Path, features: Path, run: Path) -> None:
    source_dir = checkpoint.parent
    setting = TUNED_SETTING
    rows = []
    for task in FEFF_TASKS:
        setting_name = f"{setting['name']}_seed{setting['seed']}"
        out = run / "heads" / "tunedUniversalXAS" / task / setting_name
        tuned_checkpoint = best_checkpoint(out)
        if tuned_checkpoint is None:
            ensure_clean_or_complete(out, f"Tuned-UniversalXAS {task}")
            pl.seed_everything(setting["seed"], workers=True)
            tuned = build_head_regressor(
                source_dir,
                lr=setting["lr"],
                dropout=setting["dropout"],
                batch=setting["batch_size"],
                epochs=setting["epochs"],
                patience=setting["patience"],
                scheduler="cosine",
                cosine_t=setting["cosine_t"],
            )
            tuned.load("best")
            tuned.cfg.directory = str(out)
            out.mkdir(parents=True, exist_ok=True)
            tuned.fit(load_feature_split(features, task))
            tuned_checkpoint = Path(tuned.cfg.fetch_checkpoint("best"))
        else:
            print(f"reusing tuned checkpoint for {task}: {tuned_checkpoint}", flush=True)
        split = load_feature_split(features, task)
        row = {
            "dataset": task,
            "setting": setting["name"],
            "seed": setting["seed"],
            "dropout": setting["dropout"],
            "lr": setting["lr"],
            "checkpoint": str(tuned_checkpoint),
            "val_loss_score": checkpoint_score(tuned_checkpoint),
        }
        row.update(evaluate_head(tuned_checkpoint, split, "val"))
        row.update(evaluate_head(tuned_checkpoint, split, "test"))
        rows.append(row)
        print(
            f"tuned {task}: val_eta={row['val_eta']:.4f} "
            f"test_eta={row['test_eta']:.4f}",
            flush=True,
        )
    csv_write(run / "tuned_eval.csv", rows)


def save_settings(
    path: Path,
    args: argparse.Namespace,
    run: Path,
    counts: dict[str, int],
    full_batches: int,
    balanced_batches: int,
    balanced_batch_size: int,
) -> None:
    scale = full_batches / balanced_batches
    settings: dict[str, Any] = {
        "args": vars(args),
        "run_dir": str(run),
        "pipeline": "balanced_e2e_feff",
        "feature_scale": FEATURE_SCALE,
        "tasks": FEFF_TASKS,
        "train_rows_by_task": counts,
        "total_train_rows": sum(counts.values()),
        "e2e_train_batch_size": balanced_batch_size,
        "eval_export_batch_size": ENCODER_BATCH,
        "balanced_batch_size": balanced_batch_size,
        "rows_per_element": args.rows_per_element,
        "full_data_batches_per_epoch": full_batches,
        "balanced_batches_per_epoch": balanced_batches,
        "update_budget_scale": scale,
        "e2e_base_plateau_patience": args.e2e_plateau_patience,
        "e2e_effective_plateau_patience": round(args.e2e_plateau_patience * scale),
        "e2e_base_early_stopping_patience": args.e2e_patience,
        "e2e_effective_early_stopping_patience": round(args.e2e_patience * scale),
        "e2e_min_lr": E2E_MIN_LR,
        "e2e_monitor": "val_balanced_rel_mse",
        "e2e_derivative_lambda": ENCODER_DERIV_LAMBDA,
        "inverse_dataset_size_loss_weights": False,
        "universal_variant": UNIVERSAL_VARIANT,
        "universal_settings": {
            "seed": UNIVERSAL_SEED,
            "dropout": UNIVERSAL_DROPOUT,
            "lr": UNIVERSAL_LR,
            "scheduler": "plateau",
            "plateau_factor": UNIVERSAL_PLATEAU_FACTOR,
            "base_plateau_patience": UNIVERSAL_PLATEAU_PATIENCE,
            "min_lr": E2E_MIN_LR,
            "epochs": UNIVERSAL_EPOCHS,
            "early_stopping_patience": UNIVERSAL_PATIENCE,
            "batch_size": HEAD_BATCH_SIZE,
            "loader": "standard_full_feature_loader",
            "monitor": "val_median_mse",
        },
        "universal_loader": "standard_full_feature_loader",
        "tuned_setting": TUNED_SETTING,
    }
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError("NumPy >=2 is incompatible with the pinned torch/MatGL stack")
    patch_matgl_gpu_constants()

    root = Path(__file__).resolve().parents[1]
    raw_root = (
        Path(os.environ.get("OMNIXAS_DATA_ROOT", root.parent / "OmniXAS_data"))
        / "materialscloud_omnixas_raw"
        / "extracted"
    )
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw data root: {raw_root}")
    output_root = Path(args.output_root)
    output_root = output_root if output_root.is_absolute() else root / output_root
    if args.resume and args.run_name is None:
        raise ValueError("--resume requires --run-name")
    run_name = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_balanced_e2e_feff_seed{args.seed}"
    run = output_root / run_name
    settings_path = run / "run_settings.json"

    if args.overwrite and run.exists():
        shutil.rmtree(run)
    if args.resume:
        if not settings_path.is_file():
            raise FileNotFoundError(f"Cannot resume run without settings: {settings_path}")
        saved = json.loads(settings_path.read_text(encoding="utf-8"))
        for key, value in saved.get("args", {}).items():
            if hasattr(args, key) and key not in {
                "resume",
                "overwrite",
                "gpu",
                "run_name",
                "output_root",
            }:
                setattr(args, key, value)
        print(f"resuming run: {run}", flush=True)
    else:
        if run.exists():
            raise FileExistsError(f"Run already exists: {run}; use --resume or --overwrite")
        run.mkdir(parents=True)

    if (run / "RUN_COMPLETE.json").is_file():
        print(f"run is already complete: {run}", flush=True)
        return

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("medium")
    validate_raw_structures(root, raw_root, FEFF_TASKS)

    train_dataset = FEFFDataset(root, raw_root, FEFF_TASKS, "train")
    train_base, val_base, counts = task_baselines(root, FEFF_TASKS)
    sampler = BalancedTaskBatchSampler(train_dataset.rows, args.rows_per_element, args.seed)
    balanced_batch_size = args.rows_per_element * len(FEFF_TASKS)
    full_batches = math.ceil(len(train_dataset) / balanced_batch_size)
    balanced_batches = len(sampler)
    if full_batches < 1 or balanced_batches < 1:
        raise ValueError("Training data must provide at least one batch")
    scale = full_batches / balanced_batches
    effective_plateau = max(1, round(args.e2e_plateau_patience * scale))
    effective_early = max(1, round(args.e2e_patience * scale))
    save_settings(
        settings_path,
        args,
        run,
        counts,
        full_batches,
        balanced_batches,
        balanced_batch_size,
    )
    print(
        f"E2E batches: batch_size={balanced_batch_size}, "
        f"full={full_batches}, balanced={balanced_batches}, "
        f"scale={scale:.3f}; "
        f"patience={effective_plateau}/{effective_early}",
        flush=True,
    )

    e2e_checkpoint = train_e2e(
        run,
        root,
        raw_root,
        args,
        train_base,
        val_base,
        train_dataset,
        effective_plateau,
        effective_early,
    )
    evaluate_e2e(e2e_checkpoint, root, raw_root, run, args)
    features = export_features(run, e2e_checkpoint, root, raw_root, args)
    universal_checkpoint = train_universal(features, e2e_checkpoint, run)
    evaluate_universal(universal_checkpoint, features, run)
    train_tuned(universal_checkpoint, features, run)
    (run / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "pipeline": "balanced_e2e_feff",
                "e2e_checkpoint": str(e2e_checkpoint),
                "universal_checkpoint": str(universal_checkpoint),
                "tuned_setting": TUNED_SETTING,
                "outputs": ["e2e_eval.csv", "universal_eval.csv", "tuned_eval.csv"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done: {run}", flush=True)


if __name__ == "__main__":
    main()
