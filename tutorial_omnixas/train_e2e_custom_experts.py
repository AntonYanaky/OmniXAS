#!/usr/bin/env python3
"""Train a scratch custom encoder with task-specific ExpertXAS heads.

Run:
    python tutorial_omnixas/train_e2e_custom_experts.py --run-name e2e_experts_seed42 --gpu 1
Resume:
    python tutorial_omnixas/train_e2e_custom_experts.py --run-name e2e_experts_seed42 --gpu 1 --resume

The encoder is trained end-to-end on all FEFF tasks. Each training sample is
routed to the ExpertXAS head for its element. After encoder training, the
encoder is frozen/exported and each preserved ExpertXAS head is continued on
its own exported feature split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
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
from torch import nn
from torch.utils.data import DataLoader, Sampler

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.xasblock import XASBlock
from omnixas.model.xasblock_regressor import XASBlockRegressor

from train_all8_feff import (
    BATCH,
    ENCODER_BATCH,
    ENCODER_DERIV_LAMBDA,
    ENCODER_PATIENCE,
    EXPORT_TASKS,
    FEATURE_SCALE,
    FEFF_TASKS,
    INPUT_DIM,
    OUTPUT_DIM,
    SPLITS,
    CollateGraphs,
    FEFFDataset,
    ScratchEncoder,
    checkpoint_score,
    csv_write,
    eval_ckpt,
    load_feature_split,
    missing_feature_splits,
    patch_matgl_gpu_constants,
    torch_load,
)

EXPERT_DIMS = {
    "Ti_FEFF": [600, 600, 450],
    "V_FEFF": [600, 550, 450],
    "Cr_FEFF": [450, 350, 150],
    "Mn_FEFF": [500, 400, 300],
    "Fe_FEFF": [450, 400, 450],
    "Co_FEFF": [600, 550, 450],
    "Ni_FEFF": [600, 300],
    "Cu_FEFF": [600, 600, 400],
}
MIN_LR = 1e-4
ETA_MIN = 1e-6


class TaskBatchSampler(Sampler[list[int]]):
    """Yield homogeneous task batches so BatchNorm heads see one task at a time."""

    def __init__(self, dataset: FEFFDataset, batch_size: int, *, shuffle: bool, drop_last: bool):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        by_task: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            by_task[row[0]].append(index)
        self.by_task = dict(by_task)

    def __iter__(self) -> Iterator[list[int]]:
        batches = []
        for indices in self.by_task.values():
            indices = list(indices)
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < max(2, self.batch_size):
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.by_task.values():
            n = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                n += 1
            total += n
        return total


class MultiExpertModel(nn.Module):
    def __init__(self, gnn_dropout: float, head_dropout: float):
        super().__init__()
        self.encoder = ScratchEncoder(gnn_dropout)
        self.task_names = list(FEFF_TASKS)
        self.task_to_idx = {task: i for i, task in enumerate(self.task_names)}
        XASBlock.DROPOUT = head_dropout
        self.heads = nn.ModuleDict({
            task: XASBlock(INPUT_DIM, EXPERT_DIMS[task], OUTPUT_DIM)
            for task in self.task_names
        })

    def forward(self, graph, site, task):
        z = self.encoder(graph, site) * FEATURE_SCALE
        out = z.new_empty(z.shape[0], OUTPUT_DIM)
        for task_name, task_idx in self.task_to_idx.items():
            mask = task == task_idx
            if mask.any():
                out[mask] = self.heads[task_name](z[mask])
        return out


class LitMultiExpert(pl.LightningModule):
    def __init__(
        self,
        model: MultiExpertModel,
        train_base: torch.Tensor,
        val_base: torch.Tensor,
        task_weights: torch.Tensor,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
        epochs: int,
    ):
        super().__init__()
        self.model = model
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.register_buffer("train_base", train_base.float())
        self.register_buffer("val_base", val_base.float())
        self.register_buffer("task_weights", task_weights.float())
        self.val_mses: list[torch.Tensor] = []
        self.val_tasks: list[torch.Tensor] = []

    def step(self, batch, stage: str):
        graph = batch["graph"].to(self.device)
        site = batch["site"].to(self.device)
        task = batch["task"].to(self.device)
        y = batch["y"].to(self.device)
        pred = self.model(graph, site, task)
        sample_mse = ((pred - y) ** 2).mean(dim=1)
        loss = ((sample_mse / self.train_base[task].clamp_min(1e-12)) * self.task_weights[task]).mean()
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
        for idx, task_name in enumerate(FEFF_TASKS):
            mask = tasks == idx
            if not mask.any():
                continue
            med = mses[mask].median()
            rel_medians.append(med / self.val_base[idx].clamp_min(1e-12))
            self.log(f"val_eta_{task_name}", self.val_base[idx] / med.clamp_min(1e-12), on_epoch=True, prog_bar=(task_name == "Cu_FEFF"))
        if rel_medians:
            self.log("val_balanced_rel_mse", torch.stack(rel_medians).mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.model.heads.parameters(), "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=ETA_MIN),
                "interval": "epoch",
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default="output/training/m3gnetAll8E2EExperts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gnn-dropout", type=float, default=0.1)
    p.add_argument("--head-dropout", type=float, default=0.5)
    p.add_argument("--e2e-epochs", type=int, default=300)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--continue-epochs", type=int, default=600)
    p.add_argument("--continue-lr", type=float, default=3e-4)
    p.add_argument("--continue-patience", type=int, default=40)
    return p.parse_args()


def baselines(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = root / "tutorial_omnixas" / "ml_data"
    train_base, val_base, counts = [], [], []
    for task in FEFF_TASKS:
        train_y = np.atleast_2d(np.loadtxt(data / f"{task}_train_y.txt", dtype=np.float32))
        counts.append(len(train_y))
        for split_name, out in (("train", train_base), ("val", val_base)):
            y = np.atleast_2d(np.loadtxt(data / f"{task}_{split_name}_y.txt", dtype=np.float32))
            base = np.repeat(train_y.mean(axis=0, keepdims=True), len(y), axis=0)
            out.append(float(np.median(np.mean((y - base) ** 2, axis=1))))
    counts_t = torch.tensor(counts, dtype=torch.float32)
    return torch.tensor(train_base), torch.tensor(val_base), counts_t.sum() / (len(FEFF_TASKS) * counts_t)


def best_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    ckpts = sorted(directory.glob("best*.ckpt"))
    return min(ckpts, key=checkpoint_score) if ckpts else None


def ensure_clean_or_complete(directory: Path, label: str) -> None:
    if best_checkpoint(directory) is None and directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Incomplete {label} exists without a best checkpoint: {directory}")


def make_loader(
    dataset: FEFFDataset,
    collate: CollateGraphs,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=TaskBatchSampler(dataset, batch_size, shuffle=shuffle, drop_last=drop_last),
        collate_fn=collate,
        num_workers=num_workers,
    )


def load_e2e_model(ckpt: Path, args: argparse.Namespace, device: torch.device) -> MultiExpertModel:
    model = MultiExpertModel(args.gnn_dropout, args.head_dropout)
    state = torch_load(ckpt).get("state_dict")
    if state is None:
        raise ValueError(f"Missing state_dict in {ckpt}")
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True)
    return model.to(device).eval()


def train_e2e(run: Path, root: Path, raw_root: Path, args: argparse.Namespace) -> Path:
    copied = run / "best_e2e_encoder_experts.ckpt"
    if copied.is_file():
        print(f"reusing end-to-end checkpoint: {copied}", flush=True)
        return copied

    train_base, val_base, weights = baselines(root)
    model = MultiExpertModel(args.gnn_dropout, args.head_dropout)
    collate = CollateGraphs(model.encoder)
    train_ds = FEFFDataset(root, raw_root, FEFF_TASKS, "train")
    val_ds = FEFFDataset(root, raw_root, FEFF_TASKS, "val")
    lit = LitMultiExpert(model, train_base, val_base, weights, args.encoder_lr, args.head_lr, args.weight_decay, args.e2e_epochs)
    ckpt = ModelCheckpoint(dirpath=run / "checkpoints", filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}", monitor="val_balanced_rel_mse", mode="min", save_top_k=1, save_last=True)
    pl.Trainer(
        max_epochs=args.e2e_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_balanced_rel_mse", patience=ENCODER_PATIENCE, mode="min"), ckpt],
        logger=CSVLogger(save_dir=str(run), name="e2e_logs"),
        log_every_n_steps=1,
    ).fit(
        lit,
        make_loader(train_ds, collate, ENCODER_BATCH, shuffle=True, drop_last=True, num_workers=args.num_workers),
        make_loader(val_ds, collate, ENCODER_BATCH, shuffle=False, drop_last=False, num_workers=args.num_workers),
    )
    if not ckpt.best_model_path:
        raise RuntimeError("End-to-end training finished without a best checkpoint")
    shutil.copy2(ckpt.best_model_path, copied)
    return copied


def export_features(run: Path, e2e_ckpt: Path, root: Path, raw_root: Path, args: argparse.Namespace) -> Path:
    features = run / "features"
    missing = missing_feature_splits(features, EXPORT_TASKS)
    if not missing:
        print("all FEFF/VASP features already present", flush=True)
        return features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_e2e_model(e2e_ckpt, args, device)
    collate = CollateGraphs(model.encoder)
    features.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for task, split_name in missing:
            loader = DataLoader(FEFFDataset(root, raw_root, [task], split_name), batch_size=ENCODER_BATCH, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
            Xs, ys = [], []
            for batch in loader:
                if task.endswith("_VASP"):
                    sizes = batch["graph"].batch_num_nodes()
                    expected = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
                    if not torch.equal(batch["site"], expected):
                        raise RuntimeError(f"VASP absorbers are not node 0 for {task} {split_name}")
                z = model.encoder(batch["graph"].to(device), batch["site"].to(device))
                Xs.append((z * FEATURE_SCALE).cpu().numpy())
                ys.append(batch["y"].numpy())
            X, y = np.concatenate(Xs), np.concatenate(ys)
            if X.shape != (len(y), INPUT_DIM) or y.shape[1] != OUTPUT_DIM:
                raise ValueError(f"Invalid export for {task} {split_name}: X={X.shape} y={y.shape}")
            np.savetxt(features / f"{task}_{split_name}_X.txt", X)
            np.savetxt(features / f"{task}_{split_name}_y.txt", y)
            print(f"exported {task} {split_name}: X={X.shape} y={y.shape}", flush=True)
    if still_missing := missing_feature_splits(features, EXPORT_TASKS):
        raise RuntimeError(f"Feature export incomplete: {still_missing}")
    return features


def expert_head_state(ckpt: Path, task: str) -> dict[str, torch.Tensor]:
    state = torch_load(ckpt).get("state_dict")
    if state is None:
        raise ValueError(f"Missing state_dict in {ckpt}")
    prefix = f"model.heads.{task}."
    out = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    if not out:
        raise ValueError(f"Missing preserved head state for {task} in {ckpt}")
    return out


def head_regressor(directory: Path, task: str, args: argparse.Namespace) -> XASBlockRegressor:
    XASBlock.DROPOUT = args.head_dropout
    return XASBlockRegressor(
        directory=str(directory),
        overwrite_save_dir=False,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=EXPERT_DIMS[task],
        batch_size=BATCH[task],
        max_epochs=args.continue_epochs,
        early_stopping_patience=args.continue_patience,
        initial_lr=args.continue_lr,
        min_lr=MIN_LR,
        use_lr_finder=False,
        use_early_stopping=True,
        monitor_metric="val_median_mse",
        shuffle=True,
        lr_scheduler="cosine",
        cosine_t_max=args.continue_epochs,
        cosine_eta_min=ETA_MIN,
    )


def continue_heads(run: Path, e2e_ckpt: Path, features: Path, args: argparse.Namespace) -> None:
    rows = []
    for task in FEFF_TASKS:
        split = load_feature_split(features, task)
        out = run / "heads" / "expertXAS" / task / "runs" / f"continued_seed{args.seed}_lr{args.continue_lr}_dropout{args.head_dropout}"
        ckpt = best_checkpoint(out)
        if ckpt is None:
            ensure_clean_or_complete(out, f"continued ExpertXAS {task}")
            reg = head_regressor(out, task, args)
            reg.model.model.load_state_dict(expert_head_state(e2e_ckpt, task), strict=True)
            reg.fit(split)
            ckpt = Path(reg.cfg.fetch_checkpoint("best"))
        else:
            print(f"reusing continued {task} head: {ckpt}", flush=True)
        element, kind = task.split("_", 1)
        row = {"element": element, "type": kind, "dataset": task, "checkpoint": str(ckpt), "val_loss_score": checkpoint_score(ckpt)}
        row.update(eval_ckpt(ckpt, split, "val", EXPERT_DIMS[task]))
        row.update(eval_ckpt(ckpt, split, "test", EXPERT_DIMS[task]))
        rows.append(row)
        print(f"continued {task}: val_eta={row['val_eta']:.4f} test_eta={row['test_eta']:.4f}", flush=True)
    csv_write(run / "continued_expert_eval.csv", rows)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(f"NumPy {np.__version__} is incompatible with torch==2.1/MatGL graph conversion. Run: pip install 'numpy<2' --force-reinstall")
    patch_matgl_gpu_constants()

    root = Path(__file__).resolve().parents[1]
    raw_root = Path(os.environ.get("OMNIXAS_DATA_ROOT", root.parent / "OmniXAS_data")) / "materialscloud_omnixas_raw" / "extracted"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw data root: {raw_root}")
    output_root = Path(args.output_root)
    output_root = output_root if output_root.is_absolute() else root / output_root
    run = output_root / (args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_e2e_experts_seed{args.seed}")
    settings = run / "run_settings.json"

    if args.overwrite and run.exists():
        shutil.rmtree(run)
    if args.resume:
        if not settings.is_file():
            raise FileNotFoundError(f"Cannot resume run without settings: {settings}")
        saved = json.loads(settings.read_text(encoding="utf-8"))["args"]
        for key, value in saved.items():
            if hasattr(args, key) and key not in {"resume", "overwrite", "gpu"}:
                setattr(args, key, value)
        print(f"resuming run: {run}", flush=True)
    else:
        if run.exists():
            raise FileExistsError(f"Run already exists: {run}; use --resume or --overwrite")
        run.mkdir(parents=True)
        settings.write_text(json.dumps({"args": vars(args) | {"run_dir": str(run)}}, indent=2), encoding="utf-8")
        print(json.dumps(vars(args) | {"run_dir": str(run)}, indent=2), flush=True)

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    e2e_ckpt = train_e2e(run, root, raw_root, args)
    features = export_features(run, e2e_ckpt, root, raw_root, args)
    continue_heads(run, e2e_ckpt, features, args)
    print("done:", run, flush=True)


if __name__ == "__main__":
    main()
