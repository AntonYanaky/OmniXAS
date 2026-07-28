#!/usr/bin/env python3
"""Train a scratch custom encoder on VASP only, then train VASP ExpertXAS heads.

Run:
    python tutorial_omnixas/train_vasp_encoder_experts.py --run-name vasp_encoder_seed42 --gpu 0
Resume:
    python tutorial_omnixas/train_vasp_encoder_experts.py --run-name vasp_encoder_seed42 --gpu 0 --resume
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
from torch.utils.data import DataLoader

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.xasblock import XASBlock
from omnixas.model.xasblock_regressor import XASBlockRegressor

from train_all8_feff import (
    ENCODER_BATCH,
    ENCODER_DERIV_LAMBDA,
    ENCODER_PATIENCE,
    FEATURE_SCALE,
    INPUT_DIM,
    OUTPUT_DIM,
    SPLITS,
    VASP_TASKS,
    CollateGraphs,
    EncoderModel,
    FEFFDataset,
    checkpoint_score,
    csv_write,
    eval_ckpt,
    load_feature_split,
    patch_matgl_gpu_constants,
    torch_load,
)

VASP_HPARAMS = {
    "Ti_VASP": {"batch_size": 64, "hidden_dims": [500, 600, 400]},
    "Cu_VASP": {"batch_size": 64, "hidden_dims": [550, 600, 450]},
}
MIN_LR = 1e-4
ETA_MIN = 1e-6


class LocalTaskCollate:
    def __init__(self, encoder):
        self.base = CollateGraphs(encoder)
        self.to_local = {8: 0, 9: 1}  # EXPORT_TASKS indices for Ti_VASP and Cu_VASP.

    def __call__(self, batch):
        out = self.base(batch)
        out["task"] = torch.tensor([self.to_local[int(i)] for i in out["task"]], dtype=torch.long)
        return out


class LitVaspEncoder(pl.LightningModule):
    def __init__(self, model, train_base, val_base, weights, lr, epochs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.register_buffer("train_base", train_base.float())
        self.register_buffer("val_base", val_base.float())
        self.register_buffer("weights", weights.float())
        self.val_mses: list[torch.Tensor] = []
        self.val_tasks: list[torch.Tensor] = []

    def step(self, batch, stage: str):
        graph = batch["graph"].to(self.device)
        site = batch["site"].to(self.device)
        task = batch["task"].to(self.device)
        y = batch["y"].to(self.device)
        pred = self.model(graph, site)
        sample_mse = ((pred - y) ** 2).mean(dim=1)
        loss = ((sample_mse / self.train_base[task].clamp_min(1e-12)) * self.weights[task]).mean()
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
        rel = []
        for i, task in enumerate(VASP_TASKS):
            mask = tasks == i
            if not mask.any():
                continue
            med = mses[mask].median()
            rel.append(med / self.val_base[i].clamp_min(1e-12))
            self.log(f"val_eta_{task}", self.val_base[i] / med.clamp_min(1e-12), on_epoch=True, prog_bar=(task == "Cu_VASP"))
        if rel:
            self.log("val_balanced_rel_mse", torch.stack(rel).mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": self.lr},
                {"params": self.model.head.parameters(), "lr": self.lr},
            ],
            weight_decay=1e-5,
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
    p.add_argument("--output-root", default="output/training/vaspOnlyEncoderExperts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--gnn-dropout", type=float, default=0.1)
    p.add_argument("--encoder-epochs", type=int, default=300)
    p.add_argument("--expert-lr", type=float, default=1e-3)
    p.add_argument("--expert-dropout", type=float, default=0.5)
    p.add_argument("--expert-epochs", type=int, default=1000)
    p.add_argument("--expert-patience", type=int, default=25)
    return p.parse_args()


def baselines(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = root / "tutorial_omnixas" / "ml_data"
    train_base, val_base, counts = [], [], []
    for task in VASP_TASKS:
        train_y = np.atleast_2d(np.loadtxt(data / f"{task}_train_y.txt", dtype=np.float32))
        counts.append(len(train_y))
        for split, out in (("train", train_base), ("val", val_base)):
            y = np.atleast_2d(np.loadtxt(data / f"{task}_{split}_y.txt", dtype=np.float32))
            base = np.repeat(train_y.mean(axis=0, keepdims=True), len(y), axis=0)
            out.append(float(np.median(np.mean((y - base) ** 2, axis=1))))
    counts_t = torch.tensor(counts, dtype=torch.float32)
    return torch.tensor(train_base), torch.tensor(val_base), counts_t.sum() / (len(VASP_TASKS) * counts_t)


def best_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    ckpts = sorted(directory.glob("best*.ckpt"))
    return min(ckpts, key=checkpoint_score) if ckpts else None


def ensure_clean_or_complete(directory: Path, label: str) -> None:
    if best_checkpoint(directory) is None and directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Incomplete {label} exists without a best checkpoint: {directory}")


def feature_complete(features: Path, task: str, split: str) -> bool:
    x_path = features / f"{task}_{split}_X.txt"
    y_path = features / f"{task}_{split}_y.txt"
    if not x_path.exists() or not y_path.exists():
        return False
    X = np.atleast_2d(np.loadtxt(x_path, dtype=np.float32))
    y = np.atleast_2d(np.loadtxt(y_path, dtype=np.float32))
    if X.shape != (len(y), INPUT_DIM) or y.shape[1] != OUTPUT_DIM:
        raise ValueError(f"Invalid {task} {split}: X={X.shape} y={y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite {task} {split} feature export")
    return True


def train_encoder(run: Path, root: Path, raw_root: Path, args: argparse.Namespace) -> Path:
    copied = run / "best_vasp_encoder.ckpt"
    if copied.is_file():
        print("encoder cached:", copied, flush=True)
        return copied

    model = EncoderModel(args.gnn_dropout)
    collate = LocalTaskCollate(model.encoder)
    train_base, val_base, weights = baselines(root)
    lit = LitVaspEncoder(model, train_base, val_base, weights, args.encoder_lr, args.encoder_epochs)
    ckpt = ModelCheckpoint(
        dirpath=run / "checkpoints",
        filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}",
        monitor="val_balanced_rel_mse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.encoder_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_balanced_rel_mse", patience=ENCODER_PATIENCE, mode="min"), ckpt],
        logger=CSVLogger(save_dir=str(run), name="encoder_logs"),
        log_every_n_steps=1,
    )
    trainer.fit(
        lit,
        DataLoader(FEFFDataset(root, raw_root, VASP_TASKS, "train"), batch_size=ENCODER_BATCH, shuffle=True, collate_fn=collate, num_workers=args.num_workers, drop_last=True),
        DataLoader(FEFFDataset(root, raw_root, VASP_TASKS, "val"), batch_size=ENCODER_BATCH, shuffle=False, collate_fn=collate, num_workers=args.num_workers),
    )
    if not ckpt.best_model_path:
        raise RuntimeError("VASP encoder training finished without a best checkpoint")
    shutil.copy2(ckpt.best_model_path, copied)
    return copied


def load_encoder(ckpt: Path, args: argparse.Namespace, device: torch.device) -> EncoderModel:
    model = EncoderModel(args.gnn_dropout)
    state = torch_load(ckpt).get("state_dict")
    if state is None:
        raise ValueError(f"Checkpoint missing state_dict: {ckpt}")
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True)
    return model.to(device).eval()


def export_features(run: Path, ckpt: Path, root: Path, raw_root: Path, args: argparse.Namespace) -> Path:
    features = run / "features"
    features.mkdir(parents=True, exist_ok=True)
    missing = [(task, split) for task in VASP_TASKS for split in SPLITS if not feature_complete(features, task, split)]
    if not missing:
        print("VASP feature splits cached", flush=True)
        return features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_encoder(ckpt, args, device)
    collate = CollateGraphs(model.encoder)
    with torch.inference_mode():
        for task, split in missing:
            loader = DataLoader(FEFFDataset(root, raw_root, [task], split), batch_size=ENCODER_BATCH, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
            Xs, ys = [], []
            for batch in loader:
                sizes = batch["graph"].batch_num_nodes()
                expected_site = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
                if not torch.equal(batch["site"], expected_site):
                    raise RuntimeError(f"VASP absorbers are not node 0 for {task} {split}")
                z = model.encoder(batch["graph"].to(device), batch["site"].to(device))
                Xs.append((z * FEATURE_SCALE).cpu().numpy())
                ys.append(batch["y"].numpy())
            X, y = np.concatenate(Xs), np.concatenate(ys)
            if X.shape != (len(y), INPUT_DIM) or y.shape[1] != OUTPUT_DIM:
                raise ValueError(f"Invalid export {task} {split}: X={X.shape} y={y.shape}")
            np.savetxt(features / f"{task}_{split}_X.txt", X)
            np.savetxt(features / f"{task}_{split}_y.txt", y)
            print(f"exported {task} {split}: X={X.shape} y={y.shape}", flush=True)
    return features


def train_experts(run: Path, features: Path, args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for task in VASP_TASKS:
        split = load_feature_split(features, task)
        h = VASP_HPARAMS[task]
        out = run / "heads" / "expertXAS" / task / "runs" / f"expert_seed{args.seed}_lr{args.expert_lr}_dropout{args.expert_dropout}"
        ckpt = best_checkpoint(out)
        if ckpt is None:
            ensure_clean_or_complete(out, f"ExpertXAS {task}")
            XASBlock.DROPOUT = args.expert_dropout
            model = XASBlockRegressor(
                directory=str(out),
                overwrite_save_dir=False,
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                hidden_dims=h["hidden_dims"],
                batch_size=h["batch_size"],
                max_epochs=args.expert_epochs,
                early_stopping_patience=args.expert_patience,
                initial_lr=args.expert_lr,
                min_lr=MIN_LR,
                use_lr_finder=True,
                use_early_stopping=True,
                monitor_metric="val_loss",
                shuffle=False,
                lr_scheduler="none",
            )
            model.fit(split)
            ckpt = Path(model.cfg.fetch_checkpoint("best"))
        else:
            print(f"expert cached for {task}: {ckpt}", flush=True)
        element, kind = task.split("_", 1)
        row = {"element": element, "type": kind, "dataset": task, "checkpoint": str(ckpt), "val_loss_score": checkpoint_score(ckpt)}
        row.update(eval_ckpt(ckpt, split, "val", h["hidden_dims"]))
        row.update(eval_ckpt(ckpt, split, "test", h["hidden_dims"]))
        rows.append(row)
        print(f"expert {task}: val_eta={row['val_eta']:.4f} test_eta={row['test_eta']:.4f}", flush=True)
    csv_write(run / "expert_eval.csv", rows)


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
    run = output_root / (args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_vasp_encoder_seed{args.seed}")
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
        print("resuming:", run, flush=True)
    else:
        if run.exists():
            raise FileExistsError(f"Run exists: {run}; use --resume or --overwrite")
        run.mkdir(parents=True)
        settings.write_text(json.dumps({"args": vars(args) | {"run_dir": str(run)}}, indent=2), encoding="utf-8")
        print(json.dumps(vars(args) | {"run_dir": str(run)}, indent=2), flush=True)

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    ckpt = train_encoder(run, root, raw_root, args)
    features = export_features(run, ckpt, root, raw_root, args)
    train_experts(run, features, args)
    print("done:", run, flush=True)


if __name__ == "__main__":
    main()
