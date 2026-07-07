#!/usr/bin/env python3
"""Evaluate UniversalXAS checkpoints trained on a custom feature directory."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.metrics import ModelMetrics
from omnixas.model.training import PlModule
from omnixas.model.xasblock import XASBlock


FEFF_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
UNIVERSAL_DIMS = [500, 500, 550]
INPUT_DIM = 64
OUTPUT_DIM = 141


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch<2.0
        return torch.load(path, map_location="cpu")


def load_split(data_dir: Path, element: str) -> MLSplits:
    task = f"{element}_FEFF"
    return MLSplits(**{
        split: MLData(
            X=np.atleast_2d(np.loadtxt(data_dir / f"{task}_{split}_X.txt", dtype=np.float32)),
            y=np.atleast_2d(np.loadtxt(data_dir / f"{task}_{split}_y.txt", dtype=np.float32)),
        )
        for split in ["train", "val", "test"]
    })


def checkpoint_val_loss(ckpt_path: Path) -> float:
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
    match = re.search(r"val_(?:loss|median_mse)[=_](\d+(?:\.\d+)?)", ckpt_path.name)
    return float(match.group(1)) if match else float("inf")


def find_best_universal(run_root: Path) -> Path:
    matches = []
    patterns = [
        run_root / "universalXAS" / "All_FEFF" / "runs" / "*" / "best*.ckpt",
        run_root / "universalXAS" / "All_FEFF" / "checkpoints" / "best*.ckpt",
    ]
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(f"No UniversalXAS checkpoints under {run_root}")
    return min(matches, key=lambda path: (checkpoint_val_loss(path), str(path)))


def predict(ckpt_path: Path, X: np.ndarray, batch_size: int) -> np.ndarray:
    module = PlModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model=XASBlock(INPUT_DIM, UNIVERSAL_DIMS, OUTPUT_DIM),
        lr=1e-4,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(module(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def eta(split: MLSplits, split_name: str, pred: np.ndarray) -> dict[str, float]:
    data = getattr(split, split_name)
    baseline = np.repeat(split.train.y.mean(axis=0, keepdims=True), len(data.y), axis=0)
    median_mse = float(ModelMetrics(predictions=pred, targets=data.y).median_of_mse_per_spectra)
    baseline_median = float(np.median(np.mean((data.y - baseline) ** 2, axis=1)))
    return {
        f"{split_name}_median_mse": median_mse,
        f"{split_name}_baseline_median_mse": baseline_median,
        f"{split_name}_eta": baseline_median / median_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-root", required=True, help="The --training-root used for train_paper_models.py")
    parser.add_argument("--elements", nargs="+", default=FEFF_ELEMENTS)
    parser.add_argument("--checkpoint", default="best", help="UniversalXAS ckpt path or 'best'")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    run_root = Path(args.run_root).resolve()
    ckpt = find_best_universal(run_root) if args.checkpoint == "best" else Path(args.checkpoint).resolve()
    rows = []
    for element in args.elements:
        split = load_split(data_dir, element)
        row = {"element": element, "checkpoint": str(ckpt), "val_loss_score": checkpoint_val_loss(ckpt)}
        for split_name in ["val", "test"]:
            pred = predict(ckpt, getattr(split, split_name).X, args.batch_size)
            row.update(eta(split, split_name, pred))
        rows.append(row)
        print(f"{element}: val_eta={row['val_eta']:.4f} test_eta={row['test_eta']:.4f}")

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print("saved:", out)


if __name__ == "__main__":
    main()
