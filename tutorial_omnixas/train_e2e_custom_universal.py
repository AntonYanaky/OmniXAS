#!/usr/bin/env python3
"""Train a scratch custom encoder end-to-end with a full UniversalXAS head.

Run:
    python tutorial_omnixas/train_e2e_custom_universal.py --run-name e2e_universal_seed42 --gpu 0
Resume:
    python tutorial_omnixas/train_e2e_custom_universal.py --run-name e2e_universal_seed42 --gpu 0 --resume

Pipeline:
1. Train scratch M3GNet-style encoder + one UniversalXAS head on FEFF+VASP.
2. Export frozen encoder features for all FEFF and VASP tasks.
3. Continue training the same UniversalXAS head on exported FEFF+VASP features.
4. Fine-tune that continued UniversalXAS on each FEFF/VASP dataset with a small
   validation-selected sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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
from torch.utils.data import DataLoader

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
    HEAD_WIDTHS,
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

MIN_LR = 1e-4
TUNED_ETA_MIN = 1e-6


class EndToEndUniversal(nn.Module):
    def __init__(self, gnn_dropout: float, head_dropout: float):
        super().__init__()
        self.encoder = ScratchEncoder(gnn_dropout)
        XASBlock.DROPOUT = head_dropout
        self.head = XASBlock(INPUT_DIM, HEAD_WIDTHS, OUTPUT_DIM)

    def forward(self, graph, site):
        return self.head(self.encoder(graph, site) * FEATURE_SCALE)


class LitEndToEnd(pl.LightningModule):
    def __init__(
        self,
        model: EndToEndUniversal,
        tasks: list[str],
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
        self.tasks = tasks
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
        pred = self.model(graph, site)
        sample_mse = ((pred - y) ** 2).mean(dim=1)
        rel = sample_mse / self.train_base[task].clamp_min(1e-12)
        loss = (rel * self.task_weights[task]).mean()
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
        for idx, task_name in enumerate(self.tasks):
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
                {"params": self.model.head.parameters(), "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=TUNED_ETA_MIN),
                "interval": "epoch",
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default="output/training/m3gnetAll8E2EUniversal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gnn-dropout", type=float, default=0.1)
    p.add_argument("--e2e-epochs", type=int, default=300)
    p.add_argument("--e2e-encoder-lr", type=float, default=1e-3)
    p.add_argument("--e2e-head-lr", type=float, default=7e-4)
    p.add_argument("--e2e-head-dropout", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--universal-lr", type=float, default=7e-4)
    p.add_argument("--universal-dropout", type=float, default=0.25)
    p.add_argument("--universal-epochs", type=int, default=800)
    p.add_argument("--universal-patience", type=int, default=60)
    p.add_argument("--tuned-epochs", type=int, default=1000)
    p.add_argument("--tuned-patience", type=int, default=25)
    return p.parse_args()


def json_ready(args: argparse.Namespace, run: Path) -> dict[str, Any]:
    return vars(args) | {"run_dir": str(run), "pipeline": "e2e_custom_encoder_universal"}


def task_baselines(root: Path, tasks: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = root / "tutorial_omnixas" / "ml_data"
    train_base, val_base, counts = [], [], []
    for task in tasks:
        train_y = np.atleast_2d(np.loadtxt(data / f"{task}_train_y.txt", dtype=np.float32))
        counts.append(len(train_y))
        for split_name, out in (("train", train_base), ("val", val_base)):
            y = np.atleast_2d(np.loadtxt(data / f"{task}_{split_name}_y.txt", dtype=np.float32))
            baseline = np.repeat(train_y.mean(axis=0, keepdims=True), len(y), axis=0)
            out.append(float(np.median(np.mean((y - baseline) ** 2, axis=1))))
    counts_t = torch.tensor(counts, dtype=torch.float32)
    weights = counts_t.sum() / (len(tasks) * counts_t)
    return torch.tensor(train_base), torch.tensor(val_base), weights


def best_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    ckpts = sorted(directory.glob("best*.ckpt"))
    return min(ckpts, key=checkpoint_score) if ckpts else None


def ensure_clean_or_complete(directory: Path, label: str) -> None:
    if best_checkpoint(directory) is None and directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Incomplete {label} exists without a best checkpoint: {directory}")


def build_head_regressor(directory: Path, *, lr: float, dropout: float, batch: int, epochs: int, patience: int, scheduler: str, cosine_t: int, warmup_epochs: int = 10) -> XASBlockRegressor:
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
        min_lr=MIN_LR,
        use_lr_finder=False,
        use_early_stopping=True,
        monitor_metric="val_median_mse",
        shuffle=True,
        lr_scheduler=scheduler,
        cosine_t_max=cosine_t,
        cosine_eta_min=TUNED_ETA_MIN,
        warmup_epochs=warmup_epochs,
    )


def head_state_from_e2e(ckpt: Path) -> dict[str, torch.Tensor]:
    state = torch_load(ckpt).get("state_dict")
    if state is None:
        raise ValueError(f"Missing state_dict in {ckpt}")
    head_state = {k.removeprefix("model.head."): v for k, v in state.items() if k.startswith("model.head.")}
    if not head_state:
        raise ValueError(f"Missing UniversalXAS head state in {ckpt}")
    return head_state


def load_e2e_model(ckpt: Path, args: argparse.Namespace, device: torch.device) -> EndToEndUniversal:
    model = EndToEndUniversal(args.gnn_dropout, args.e2e_head_dropout)
    state = torch_load(ckpt).get("state_dict")
    if state is None:
        raise ValueError(f"Missing state_dict in {ckpt}")
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True)
    return model.to(device).eval()


def export_features(run: Path, e2e_ckpt: Path, root: Path, raw_root: Path, args: argparse.Namespace) -> Path:
    features = run / "features"
    missing = missing_feature_splits(features)
    if not missing:
        print("all FEFF/VASP features already present", flush=True)
        return features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_e2e_model(e2e_ckpt, args, device)
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
            Xs, ys = [], []
            for batch in loader:
                if task.endswith("_VASP"):
                    sizes = batch["graph"].batch_num_nodes()
                    expected_sites = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
                    if not torch.equal(batch["site"], expected_sites):
                        raise RuntimeError(f"VASP absorbers are not node 0 for {task} {split_name}")
                z = model.encoder(batch["graph"].to(device), batch["site"].to(device))
                Xs.append((z * FEATURE_SCALE).cpu().numpy())
                ys.append(batch["y"].numpy())
            X = np.concatenate(Xs)
            y = np.concatenate(ys)
            if X.shape != (len(y), INPUT_DIM) or y.shape[1] != OUTPUT_DIM:
                raise ValueError(f"Invalid export for {task} {split_name}: X={X.shape} y={y.shape}")
            np.savetxt(features / f"{task}_{split_name}_X.txt", X)
            np.savetxt(features / f"{task}_{split_name}_y.txt", y)
            print(f"exported {task} {split_name}: X={X.shape} y={y.shape}", flush=True)
    if still_missing := missing_feature_splits(features):
        raise RuntimeError(f"Feature export incomplete: {still_missing}")
    return features


def universal_split(features: Path) -> MLSplits:
    splits = [load_feature_split(features, task) for task in EXPORT_TASKS]
    return MLSplits(
        train=MLData(X=np.concatenate([s.train.X for s in splits]), y=np.concatenate([s.train.y for s in splits])),
        val=MLData(X=np.concatenate([s.val.X for s in splits]), y=np.concatenate([s.val.y for s in splits])),
        test=MLData(X=np.concatenate([s.test.X for s in splits]), y=np.concatenate([s.test.y for s in splits])),
    )


def tuned_sweep(task: str) -> list[dict[str, Any]]:
    if task == "Ti_VASP":
        return [
            {"name": "warmcosT600_lr5e-4_do0p1_w20", "lr": 5e-4, "dropout": 0.10, "scheduler": "warmup_cosine", "cosine_t": 600, "warmup": 20},
            {"name": "cosT600_lr5e-4_do0p25", "lr": 5e-4, "dropout": 0.25, "scheduler": "cosine", "cosine_t": 600, "warmup": 0},
            {"name": "cosT600_lr4e-4_do0p25", "lr": 4e-4, "dropout": 0.25, "scheduler": "cosine", "cosine_t": 600, "warmup": 0},
        ]
    if task == "Cu_VASP":
        return [
            {"name": "cosT600_lr5e-4_do0p25", "lr": 5e-4, "dropout": 0.25, "scheduler": "cosine", "cosine_t": 600, "warmup": 0},
            {"name": "cosT600_lr4e-4_do0p25", "lr": 4e-4, "dropout": 0.25, "scheduler": "cosine", "cosine_t": 600, "warmup": 0},
            {"name": "cosT600_lr4e-4_do0p15", "lr": 4e-4, "dropout": 0.15, "scheduler": "cosine", "cosine_t": 600, "warmup": 0},
        ]
    return [
        {"name": "cosT250_lr1e-4_do0p0", "lr": 1e-4, "dropout": 0.00, "scheduler": "cosine", "cosine_t": 250, "warmup": 0},
        {"name": "cosT250_lr1e-4_do0p1", "lr": 1e-4, "dropout": 0.10, "scheduler": "cosine", "cosine_t": 250, "warmup": 0},
        {"name": "cosT500_lr3e-4_do0p1", "lr": 3e-4, "dropout": 0.10, "scheduler": "cosine", "cosine_t": 500, "warmup": 0},
    ]


def train_e2e(run: Path, root: Path, raw_root: Path, args: argparse.Namespace) -> Path:
    copied = run / "best_e2e_encoder_universal.ckpt"
    if copied.is_file():
        print(f"reusing end-to-end checkpoint: {copied}", flush=True)
        return copied

    tasks = list(EXPORT_TASKS)
    train_base, val_base, weights = task_baselines(root, tasks)
    model = EndToEndUniversal(args.gnn_dropout, args.e2e_head_dropout)
    collate = CollateGraphs(model.encoder)
    train_loader = DataLoader(FEFFDataset(root, raw_root, tasks, "train"), batch_size=ENCODER_BATCH, shuffle=True, collate_fn=collate, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(FEFFDataset(root, raw_root, tasks, "val"), batch_size=ENCODER_BATCH, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
    lit = LitEndToEnd(model, tasks, train_base, val_base, weights, args.e2e_encoder_lr, args.e2e_head_lr, args.weight_decay, args.e2e_epochs)
    ckpt = ModelCheckpoint(dirpath=run / "checkpoints", filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}", monitor="val_balanced_rel_mse", mode="min", save_top_k=1, save_last=True)
    pl.Trainer(
        max_epochs=args.e2e_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_balanced_rel_mse", patience=ENCODER_PATIENCE, mode="min"), ckpt],
        logger=CSVLogger(save_dir=str(run), name="e2e_logs"),
        log_every_n_steps=1,
    ).fit(lit, train_loader, val_loader)
    if not ckpt.best_model_path:
        raise RuntimeError("End-to-end training finished without a best checkpoint")
    shutil.copy2(ckpt.best_model_path, copied)
    return copied


def continue_universal(run: Path, e2e_ckpt: Path, features: Path, args: argparse.Namespace) -> Path:
    out = run / "heads" / "universalXAS" / "All_FEFF_VASP" / "runs" / f"continued_seed{args.seed}_lr{args.universal_lr}_dropout{args.universal_dropout}"
    ckpt = best_checkpoint(out)
    if ckpt is not None:
        print(f"reusing continued UniversalXAS checkpoint: {ckpt}", flush=True)
        return ckpt
    ensure_clean_or_complete(out, "continued UniversalXAS")
    reg = build_head_regressor(
        out,
        lr=args.universal_lr,
        dropout=args.universal_dropout,
        batch=32,
        epochs=args.universal_epochs,
        patience=args.universal_patience,
        scheduler="cosine",
        cosine_t=args.universal_epochs,
    )
    reg.model.model.load_state_dict(head_state_from_e2e(e2e_ckpt), strict=True)
    reg.fit(universal_split(features))
    return Path(reg.cfg.fetch_checkpoint("best"))


def evaluate_universal(ckpt: Path, features: Path, run: Path) -> None:
    rows = []
    for task in EXPORT_TASKS:
        element, kind = task.split("_", 1)
        split = load_feature_split(features, task)
        row = {"element": element, "type": kind, "dataset": task, "checkpoint": str(ckpt), "val_loss_score": checkpoint_score(ckpt)}
        row.update(eval_ckpt(ckpt, split, "val"))
        row.update(eval_ckpt(ckpt, split, "test"))
        rows.append(row)
        print(f"universal {task}: val_eta={row['val_eta']:.4f} test_eta={row['test_eta']:.4f}", flush=True)
    csv_write(run / "universal_eval.csv", rows)


def tune_all(continued_ckpt: Path, features: Path, run: Path, args: argparse.Namespace) -> None:
    source_dir = continued_ckpt.parent
    candidate_rows = []
    selected_rows = []
    for task in EXPORT_TASKS:
        element, kind = task.split("_", 1)
        split = load_feature_split(features, task)
        task_rows = []
        for cfg in tuned_sweep(task):
            out = run / "heads" / "tunedUniversalXAS" / task / "runs" / f"{cfg['name']}_seed{args.seed}"
            ckpt = best_checkpoint(out)
            if ckpt is None:
                ensure_clean_or_complete(out, f"tuned {task} {cfg['name']}")
                reg = build_head_regressor(
                    source_dir,
                    lr=cfg["lr"],
                    dropout=cfg["dropout"],
                    batch=BATCH[task],
                    epochs=args.tuned_epochs,
                    patience=args.tuned_patience,
                    scheduler=cfg["scheduler"],
                    cosine_t=cfg["cosine_t"],
                    warmup_epochs=cfg["warmup"],
                )
                reg.load("best")
                reg.cfg.directory = str(out)
                out.mkdir(parents=True, exist_ok=True)
                reg.fit(split)
                ckpt = Path(reg.cfg.fetch_checkpoint("best"))
            else:
                print(f"reusing tuned checkpoint for {task} {cfg['name']}: {ckpt}", flush=True)
            row = {"element": element, "type": kind, "dataset": task, "setting": cfg["name"], "dropout": cfg["dropout"], "lr": cfg["lr"], "checkpoint": str(ckpt), "val_loss_score": checkpoint_score(ckpt)}
            row.update(eval_ckpt(ckpt, split, "val"))
            candidate_rows.append(row)
            task_rows.append(row)
            print(f"tuned {task} {cfg['name']}: val_eta={row['val_eta']:.4f}", flush=True)
        selected = dict(max(task_rows, key=lambda r: r["val_eta"]))
        selected.update(eval_ckpt(Path(selected["checkpoint"]), split, "test"))
        selected_rows.append(selected)
        print(f"selected {task}: val_eta={selected['val_eta']:.4f} test_eta={selected['test_eta']:.4f}", flush=True)
    csv_write(run / "tuned_validation_candidates.csv", candidate_rows)
    csv_write(run / "tuned_eval.csv", selected_rows)


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
    run = output_root / (args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_e2e_universal_seed{args.seed}")
    settings_path = run / "run_settings.json"

    if args.overwrite and run.exists():
        shutil.rmtree(run)
    if args.resume:
        if not settings_path.is_file():
            raise FileNotFoundError(f"Cannot resume run without settings: {settings_path}")
        saved = json.loads(settings_path.read_text(encoding="utf-8"))
        for key, value in saved.get("args", saved).items():
            if hasattr(args, key) and key not in {"resume", "overwrite", "gpu"}:
                setattr(args, key, value)
        print(f"resuming run: {run}", flush=True)
    else:
        if run.exists():
            raise FileExistsError(f"Run already exists: {run}; use --resume or --overwrite")
        run.mkdir(parents=True)
        settings_path.write_text(json.dumps({"args": json_ready(args, run)}, indent=2), encoding="utf-8")
        print(json.dumps(json_ready(args, run), indent=2), flush=True)

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    e2e_ckpt = train_e2e(run, root, raw_root, args)
    features = export_features(run, e2e_ckpt, root, raw_root, args)
    continued_ckpt = continue_universal(run, e2e_ckpt, features, args)
    evaluate_universal(continued_ckpt, features, run)
    tune_all(continued_ckpt, features, run, args)
    print("done:", run, flush=True)


if __name__ == "__main__":
    main()
