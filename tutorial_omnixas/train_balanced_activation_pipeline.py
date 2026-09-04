#!/usr/bin/env python3
"""Train one complete balanced FEFF pipeline for one activation variant.

Use separate variant directories for simultaneous different variants. Do not run
more than one process for the same variant at the same time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterator

_boot = argparse.ArgumentParser(add_help=False)
_boot.add_argument("--gpu")
_boot_args, _ = _boot.parse_known_args()
if _boot_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _boot_args.gpu

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from train_all8_feff import (  # noqa: E402
    CollateGraphs, ENCODER_BATCH, FEFFDataset, ScratchEncoder,
    patch_matgl_gpu_constants, torch_load,
)

TASKS = [f"{element}_FEFF" for element in "Ti V Cr Mn Fe Co Ni Cu".split()]
ELEMENTS = [task.split("_")[0] for task in TASKS]
SPLITS = ("train", "val", "test")
INPUT_DIM, OUTPUT_DIM, SCALE = 64, 141, 1000.0
WIDTHS = (500, 500, 550)


def raw_data_root() -> Path:
    path = Path(os.environ.get("OMNIXAS_DATA_ROOT", ROOT.parent / "OmniXAS_data"))
    path = path / "materialscloud_omnixas_raw" / "extracted"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing raw data root: {path}")
    return path


def canonical_rows(task: str, split: str) -> tuple[list[str], np.ndarray]:
    ids_path = ROOT / "tutorial_omnixas" / "material_id_and_site" / f"{task}_{split}.txt"
    y_path = ROOT / "tutorial_omnixas" / "ml_data" / f"{task}_{split}_y.txt"
    if not ids_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Missing required data for {task} {split}")
    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    y = np.atleast_2d(np.loadtxt(y_path, dtype=np.float32))
    if y.shape != (len(ids), OUTPUT_DIM) or not np.isfinite(y).all():
        raise ValueError(f"Invalid target shape or value for {task} {split}: {y.shape}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs for {task} {split}")
    return ids, y


def validate_data() -> None:
    for task in TASKS:
        split_ids = []
        for split in SPLITS:
            ids, _ = canonical_rows(task, split)
            split_ids.append({row.rsplit("_", 1)[0] for row in ids})
        if set.intersection(*split_ids):
            raise ValueError(f"Material occurs in multiple splits for {task}")


def baselines() -> tuple[torch.Tensor, torch.Tensor]:
    train, val = [], []
    for task in TASKS:
        _, train_y = canonical_rows(task, "train")
        _, val_y = canonical_rows(task, "val")
        mean = train_y.mean(axis=0)
        train.append(np.median(np.mean((train_y - mean) ** 2, axis=1)))
        val.append(np.median(np.mean((val_y - mean) ** 2, axis=1)))
    return torch.tensor(train), torch.tensor(val)


class BalancedSampler(Sampler[list[int]]):
    """Draw exact balanced batches without replacement, then redraw each epoch."""
    def __init__(self, task_ids: list[str], batch_size: int, seed: int):
        if batch_size < 8 or batch_size % 8:
            raise ValueError("Balanced batch size must be positive and divisible by eight")
        self.by_task = {task: [i for i, value in enumerate(task_ids) if value == task] for task in TASKS}
        self.per = batch_size // 8
        if min(len(rows) for rows in self.by_task.values()) < self.per:
            raise ValueError("A balanced batch does not fit every training task")
        self.count = min(len(rows) for rows in self.by_task.values()) // self.per
        self.seed, self.epoch = seed, 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        chosen = {task: rng.permutation(rows)[: self.count * self.per] for task, rows in self.by_task.items()}
        for number in range(self.count):
            batch = np.concatenate([chosen[task][number * self.per:(number + 1) * self.per] for task in TASKS])
            rng.shuffle(batch)
            yield batch.tolist()


class Head(nn.Sequential):
    def __init__(self, variant: str, dropout: float = .10):
        layers: list[nn.Module] = []
        dims = [INPUT_DIM, *WIDTHS, OUTPUT_DIM]
        for index, (left, right) in enumerate(zip(dims, dims[1:])):
            linear = nn.Linear(left, right)
            if variant == "selu":
                with torch.no_grad():
                    linear.weight.normal_(0.0, 1.0 / math.sqrt(left))
                    linear.bias.zero_()
            layers.append(linear)
            if index < len(WIDTHS):
                if variant == "selu":
                    layers.extend([nn.SELU(), nn.AlphaDropout(dropout)])
                else:
                    activation = nn.SiLU() if variant == "silu" else nn.GELU()
                    layers.extend([nn.BatchNorm1d(right), activation, nn.Dropout(dropout)])
            else:
                layers.append(nn.Softplus())
        super().__init__(*layers)


class E2E(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.encoder = ScratchEncoder(.10)
        self.head = Head(variant)
        if variant == "selu":
            self.register_buffer("feature_mean", torch.zeros(INPUT_DIM))
            self.register_buffer("feature_std", torch.ones(INPUT_DIM))

    def transform_features(self, encoder_output: torch.Tensor) -> torch.Tensor:
        features = encoder_output * SCALE
        if self.variant == "selu":
            features = (features - self.feature_mean) / self.feature_std
        return features

    def forward(self, graph, site):
        return self.head(self.transform_features(self.encoder(graph, site)))

    def rebase_features(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        with torch.no_grad():
            self.head[0].bias.add_(self.head[0].weight @ mean)
            self.head[0].weight.mul_(std)
            self.feature_mean.add_(self.feature_std * mean)
            self.feature_std.mul_(std)


def training_feature_stats(model: E2E, train: FEFFDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure transformed features from the training data."""
    if model.variant != "selu":
        raise ValueError("Feature standardization is only defined for SELU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collate = CollateGraphs(model.encoder)
    loader = DataLoader(train, batch_size=ENCODER_BATCH, collate_fn=collate)
    was_training = model.encoder.training
    model.encoder.eval()
    try:
        with torch.inference_mode():
            values = [model.transform_features(model.encoder(
                batch["graph"].to(device), batch["site"].to(device))).cpu() for batch in loader]
    finally:
        model.encoder.train(was_training)
        model.cpu()
    if not values:
        raise ValueError("Training feature dataset is empty")
    features = torch.cat(values)
    if features.ndim != 2 or features.shape[1] != INPUT_DIM or not torch.isfinite(features).all():
        raise ValueError("Training features are non-finite or have an invalid shape")
    mean, std = features.mean(0), features.std(0, unbiased=False)
    if not torch.isfinite(std).all() or (std <= 1e-8).any():
        raise ValueError("Training feature standard deviation is invalid")
    return mean, std


class E2ELightning(pl.LightningModule):
    def __init__(self, model: E2E, train_base: torch.Tensor, val_base: torch.Tensor, plateau_patience: int):
        super().__init__()
        self.model, self.plateau_patience = model, plateau_patience
        self.register_buffer("train_base", train_base.float())
        self.register_buffer("val_base", val_base.float())
        self.values: list[torch.Tensor] = []
        self.tasks: list[torch.Tensor] = []

    def step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pred = self.model(batch["graph"].to(self.device), batch["site"].to(self.device))
        y, task = batch["y"].to(self.device), batch["task"].to(self.device)
        mse = (pred - y).square().mean(1)
        base = self.train_base if stage == "train" else self.val_base
        loss = (mse / base[task].clamp_min(1e-12)).mean()
        loss = loss + .02 * (torch.diff(pred, dim=1) - torch.diff(y, dim=1)).square().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True)
        if stage == "val":
            self.values.append(mse.detach())
            self.tasks.append(task.detach())
        return loss

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def on_validation_epoch_start(self):
        self.values, self.tasks = [], []

    def validation_step(self, batch, _):
        return self.step(batch, "val")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        values, tasks = torch.cat(self.values), torch.cat(self.tasks)
        metric = []
        for index in range(8):
            selected = values[tasks == index]
            if not len(selected):
                raise ValueError(f"Validation has no rows for element index {index}")
            metric.append(selected.median() / self.val_base[index].clamp_min(1e-12))
        self.log("val_balanced_rel_mse", torch.stack(metric).mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.model.encoder.parameters(), "lr": 1e-3},
            {"params": self.model.head.parameters(), "lr": 5e-4},
        ], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=.5, patience=self.plateau_patience, min_lr=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler, "interval": "epoch", "frequency": 2,
            "monitor": "val_balanced_rel_mse",
        }}


class HeadLightning(pl.LightningModule):
    def __init__(self, head: Head, lr: float, universal: bool, baselines: torch.Tensor | None,
                 plateau_patience: int = 8, early_patience: int = 60):
        super().__init__()
        self.head, self.lr, self.universal = head, lr, universal
        self.plateau_patience, self.early_patience = plateau_patience, early_patience
        if baselines is not None:
            self.register_buffer("baselines", baselines.float())
        self.values: list[torch.Tensor] = []
        self.tasks: list[torch.Tensor] = []

    def training_step(self, batch, _):
        return (self.head(batch[0]) - batch[1]).square().mean()

    def on_validation_epoch_start(self):
        self.values, self.tasks = [], []

    def validation_step(self, batch, _):
        self.values.append((self.head(batch[0]) - batch[1]).square().mean(1).detach())
        if len(batch) > 2:
            self.tasks.append(batch[2].detach())

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        values = torch.cat(self.values)
        if self.universal:
            tasks = torch.cat(self.tasks)
            metrics = []
            for index in range(8):
                selected = values[tasks == index]
                if not len(selected):
                    raise ValueError(f"Universal validation has no rows for element index {index}")
                metrics.append(selected.median() / self.baselines[index].clamp_min(1e-12))
            value, name = torch.stack(metrics).mean(), "val_balanced_rel_mse"
        else:
            value, name = values.median(), "val_median_mse"
        self.log(name, value, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.universal:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=.5, patience=self.plateau_patience, min_lr=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch", "frequency": 2,
                "monitor": "val_balanced_rel_mse",
            }}
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=500, eta_min=1e-6), "interval": "epoch",
        }}


def done(job: Path, expected: list[str]) -> bool:
    if (job / "DONE").is_file() and all((job / name).is_file() for name in expected):
        return True
    if job.exists() and any(job.iterdir()):
        raise RuntimeError(f"Incomplete job directory without DONE: {job}. Remove it before rerun.")
    return False


def mark_done(job: Path) -> None:
    (job / "DONE").write_text("done\n", encoding="utf8")


def train_e2e(run: Path, raw: Path, variant: str, train_base: torch.Tensor, val_base: torch.Tensor) -> Path:
    job = run / "e2e"
    best = job / "best.ckpt"
    if done(job, ["best.ckpt", "last.ckpt"]):
        return best
    job.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(44, workers=True)
    model = E2E(variant)
    train = FEFFDataset(ROOT, raw, TASKS, "train")
    if variant == "selu":
        feature_mean, feature_std = training_feature_stats(model, train)
        model.feature_mean.copy_(feature_mean)
        model.feature_std.copy_(feature_std)
    collate = CollateGraphs(model.encoder)
    sampler = BalancedSampler([row[0] for row in train.rows], 48, 44)
    full_updates = math.ceil(len(train) / 48)
    scale = full_updates / len(sampler)
    loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate)
    validation = DataLoader(FEFFDataset(ROOT, raw, TASKS, "val"), batch_size=ENCODER_BATCH, collate_fn=collate)
    callback = ModelCheckpoint(job, filename="best", monitor="val_balanced_rel_mse", mode="min", save_top_k=1, save_last=True)
    trainer = pl.Trainer(max_epochs=1000, accelerator="auto", devices=1, check_val_every_n_epoch=1,
                         callbacks=[callback, EarlyStopping(monitor="val_balanced_rel_mse", mode="min", patience=max(1, round(60 * scale)))],
                         logger=CSVLogger(str(job), name="logs"), num_sanity_val_steps=0)
    trainer.fit(E2ELightning(model, train_base, val_base, max(1, round(8 * scale))), loader, validation)
    if not callback.best_model_path or not best.is_file():
        raise RuntimeError("E2E training did not produce a best checkpoint")
    if variant == "selu":
        checkpoint = torch_load(best)
        state = checkpoint.get("state_dict", {})
        model = E2E(variant)
        model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()
                               if key.startswith("model.")}, strict=True)
        mean, std = training_feature_stats(model, train)
        model.rebase_features(mean, std)
        checkpoint["state_dict"].update({f"model.{key}": value for key, value in model.state_dict().items()})
        torch.save(checkpoint, best)
    mark_done(job)
    return best


def load_head_state(checkpoint: Path, prefix: str) -> dict[str, torch.Tensor]:
    state = torch_load(checkpoint).get("state_dict", {})
    selected = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"Missing head state in {checkpoint}")
    return selected


def export_features(run: Path, raw: Path, e2e: Path, variant: str) -> Path:
    out = run / "features"
    names = [f"{task}_{split}_{suffix}.txt" for task in TASKS for split in SPLITS for suffix in ("X", "y")]
    if done(out, names):
        for task in TASKS:
            for split in SPLITS:
                load_features(out, task, split)
        return out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = E2E(variant)
    state = torch_load(e2e).get("state_dict", {})
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}, strict=True)
    model = model.to(device).eval()
    collate = CollateGraphs(model.encoder)
    with torch.inference_mode():
        for task in TASKS:
            for split in SPLITS:
                loader = DataLoader(FEFFDataset(ROOT, raw, [task], split), batch_size=ENCODER_BATCH, collate_fn=collate)
                xs, ys = [], []
                for batch in loader:
                    xs.append(model.transform_features(
                        model.encoder(batch["graph"].to(device), batch["site"].to(device))).cpu().numpy())
                    ys.append(batch["y"].numpy())
                X, exported_y = np.concatenate(xs), np.concatenate(ys)
                _, y = canonical_rows(task, split)
                if X.shape != (len(y), INPUT_DIM) or not np.isfinite(X).all() or not np.array_equal(exported_y, y):
                    raise ValueError(f"Invalid exported features or targets for {task} {split}")
                np.savetxt(out / f"{task}_{split}_X.txt", X)
                np.savetxt(out / f"{task}_{split}_y.txt", y)
    for task in TASKS:
        for split in SPLITS:
            load_features(out, task, split)
    mark_done(out)
    return out


def load_features(directory: Path, task: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    ids, canonical = canonical_rows(task, split)
    X = np.atleast_2d(np.loadtxt(directory / f"{task}_{split}_X.txt", dtype=np.float32))
    y = np.atleast_2d(np.loadtxt(directory / f"{task}_{split}_y.txt", dtype=np.float32))
    if X.shape != (len(ids), INPUT_DIM) or y.shape != canonical.shape or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Invalid exported split for {task} {split}")
    if not np.array_equal(y, canonical):
        raise ValueError(f"Exported targets do not exactly match canonical targets: {task} {split}")
    return X, y


def train_tuned(job: Path, variant: str, train_x: np.ndarray, train_y: np.ndarray,
                val_x: np.ndarray, val_y: np.ndarray, seed: int, batch_size: int,
                initial: dict[str, torch.Tensor]) -> None:
    if done(job, ["best.ckpt", "last.ckpt"]):
        return
    job.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed, workers=True)
    head = Head(variant)
    head.load_state_dict(initial, strict=True)
    callback = ModelCheckpoint(job, filename="best", monitor="val_median_mse", mode="min", save_top_k=1, save_last=True)
    trainer = pl.Trainer(max_epochs=1000, accelerator="auto", devices=1,
        callbacks=[callback, EarlyStopping(monitor="val_median_mse", mode="min", patience=60)],
        logger=CSVLogger(str(job), name="logs"), num_sanity_val_steps=0)
    train = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    trainer.fit(HeadLightning(head, 3e-4, False, None), DataLoader(train, batch_size=batch_size, shuffle=True),
                DataLoader(val, batch_size=ENCODER_BATCH))
    if not callback.best_model_path:
        raise RuntimeError(f"Tuned training did not produce a best checkpoint: {job}")
    mark_done(job)


def universal_and_tuned(run: Path, features: Path, e2e: Path, variant: str) -> None:
    train_parts = [load_features(features, task, "train") for task in TASKS]
    val_parts = [load_features(features, task, "val") for task in TASKS]
    train_x = np.concatenate([part[0] for part in train_parts])
    train_y = np.concatenate([part[1] for part in train_parts])
    val_x = np.concatenate([part[0] for part in val_parts])
    val_y = np.concatenate([part[1] for part in val_parts])
    train_ids = [task for task, part in zip(TASKS, train_parts) for _ in range(len(part[0]))]
    val_tasks = np.concatenate([np.full(len(part[0]), index) for index, part in enumerate(val_parts)])
    initial = load_head_state(e2e, "model.head.")
    for size in (64, 128, 192):
        for seed in (44, 45, 46):
            job = run / "universal" / f"batch_{size}" / f"seed_{seed}"
            if not done(job, ["best.ckpt", "last.ckpt"]):
                pl.seed_everything(seed, workers=True)
                sampler = BalancedSampler(train_ids, size, seed)
                full = math.ceil(len(train_y) / size)
                scale = full / len(sampler)
                loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_sampler=sampler)
                callback = ModelCheckpoint(job, filename="best", monitor="val_balanced_rel_mse", mode="min", save_top_k=1, save_last=True)
                early_patience = max(1, round(60 * scale))
                plateau_patience = max(1, round(8 * scale))
                trainer = pl.Trainer(max_epochs=800, check_val_every_n_epoch=2, accelerator="auto", devices=1,
                    callbacks=[callback, EarlyStopping(monitor="val_balanced_rel_mse", mode="min", patience=early_patience)],
                    logger=CSVLogger(str(job), name="logs"), num_sanity_val_steps=0)
                universal_head = Head(variant)
                universal_head.load_state_dict(initial, strict=True)
                validation_baselines = torch.tensor([np.median(np.mean((p[1] - q[1].mean(0)) ** 2, 1)) for q, p in zip(train_parts, val_parts)])
                trainer.fit(HeadLightning(universal_head, 5e-4, True, validation_baselines, plateau_patience, early_patience),
                            loader, DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y), torch.from_numpy(val_tasks)), batch_size=ENCODER_BATCH))
                if not callback.best_model_path:
                    raise RuntimeError(f"Universal training did not produce a best checkpoint: {job}")
                mark_done(job)
            source = job / "best.ckpt"
            source_state = load_head_state(source, "head.")
            for element_index, element in enumerate(ELEMENTS):
                tx, ty = train_parts[element_index]
                vx, vy = val_parts[element_index]
                for tuned_seed in (145, 146, 147):
                    tuned = run / "tuned" / f"source_batch_{size}" / f"source_seed_{seed}" / element / f"seed_{tuned_seed}"
                    train_tuned(tuned, variant, tx, ty, vx, vy, tuned_seed, size, source_state)


def settings(variant: str) -> dict[str, Any]:
    result = {"variant": variant, "dimensions": [64, 500, 500, 550, 141], "feature_scale": 1000.0,
            "dropout": .10, "e2e": {"seed": 44, "batch_size": 48, "rows_per_element": 6, "encoder_lr": 1e-3, "head_lr": 5e-4, "weight_decay": 1e-5, "derivative_loss": .02, "max_epochs": 1000, "early_stop": 60, "plateau_factor": .5, "plateau_patience": 8, "plateau_frequency": 2, "min_lr": 1e-6},
            "universal": {"batch_sizes": [64, 128, 192], "seeds": [44, 45, 46], "lr": 5e-4, "max_epochs": 800, "early_stop": 60, "plateau_factor": .5, "plateau_patience": 8, "frequency": 2, "min_lr": 1e-6, "validation_frequency": 2},
            "tuned": {"source_seeds": [44, 45, 46], "seeds": [145, 146, 147], "lr": 3e-4, "max_epochs": 1000, "early_stop": 60, "cosine_t_max": 500, "min_lr": 1e-6}}
    if variant == "selu":
        result["selu_recipe"] = {
            "feature_standardization": "train_only_initial_and_final_rebase",
            "initialization": "lecun_normal",
            "dropout": "alpha_dropout",
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("silu", "gelu", "selu"), required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--output-root", default="output/training/balanced_activation_pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError("NumPy >=2 is incompatible with the pinned torch/MatGL stack")
    patch_matgl_gpu_constants()
    raw = raw_data_root()
    validate_data()
    run = Path(args.output_root)
    if not run.is_absolute():
        run = ROOT / run
    run = run / args.variant
    run.mkdir(parents=True, exist_ok=True)
    fixed = settings(args.variant)
    settings_path = run / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != fixed:
            raise RuntimeError(f"Existing settings differ: {settings_path}")
    elif any(run.iterdir()):
        raise RuntimeError(f"Non-empty variant directory lacks settings.json: {run}")
    else:
        settings_path.write_text(json.dumps(fixed, indent=2) + "\n")
    train_base, val_base = baselines()
    e2e = train_e2e(run, raw, args.variant, train_base, val_base)
    features = export_features(run, raw, e2e, args.variant)
    universal_and_tuned(run, features, e2e, args.variant)
    print(f"Completed variant pipeline: {run}")


if __name__ == "__main__":
    main()
