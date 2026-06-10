#!/usr/bin/env python3
"""Find the best Ti paper checkpoints by eta across output/training.

This script intentionally cherry-picks by the reported test-set eta. It is for
finding the single best available model in your output folders.
"""

from datetime import datetime
from pathlib import Path
import argparse
import re
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.metrics import ModelMetrics
from omnixas.model.xasblock_regressor import XASBlockRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tutorial_omnixas" / "ml_data"
OUTPUT_ROOT = ROOT / "output" / "training"
RESULTS_DIR = OUTPUT_ROOT / "comparisons" / "best_eta_ti" / datetime.now().strftime("%Y%m%d_%H%M%S")

INPUT_DIM, OUTPUT_DIM = 64, 141
UNIVERSAL_DIMS = [500, 500, 550]
TI_FEFF_EXPERT_DIMS = [600, 600, 450]
TI_VASP_EXPERT_DIMS = [500, 600, 400]

PAPER_ETA = {
    ("Ti FEFF", "ExpertXAS"): 6.35,
    ("Ti FEFF", "UniversalXAS"): 4.19,
    ("Ti FEFF", "Tuned-UniversalXAS"): 7.63,
    ("Ti VASP", "ExpertXAS"): 4.75,
    ("Ti VASP", "Tuned-UniversalXAS"): 5.27,
}

MODEL_SPECS = [
    {
        "dataset": "Ti FEFF",
        "model": "ExpertXAS",
        "root": OUTPUT_ROOT / "expertXAS" / "Ti_FEFF" / "runs",
        "hidden_dims": TI_FEFF_EXPERT_DIMS,
        "split": ("Ti", "FEFF"),
        "batch_size": 32,
    },
    {
        "dataset": "Ti FEFF",
        "model": "UniversalXAS",
        "root": OUTPUT_ROOT / "universalXAS" / "All_FEFF" / "runs",
        "hidden_dims": UNIVERSAL_DIMS,
        "split": ("Ti", "FEFF"),
        "batch_size": 32,
    },
    {
        "dataset": "Ti FEFF",
        "model": "Tuned-UniversalXAS",
        "root": OUTPUT_ROOT / "tunedUniversalXAS" / "Ti_FEFF" / "runs",
        "hidden_dims": UNIVERSAL_DIMS,
        "split": ("Ti", "FEFF"),
        "batch_size": 32,
    },
    {
        "dataset": "Ti VASP",
        "model": "ExpertXAS",
        "root": OUTPUT_ROOT / "expertXAS" / "Ti_VASP" / "runs",
        "hidden_dims": TI_VASP_EXPERT_DIMS,
        "split": ("Ti", "VASP"),
        "batch_size": 64,
    },
    {
        "dataset": "Ti VASP",
        "model": "Tuned-UniversalXAS",
        "root": OUTPUT_ROOT / "tunedUniversalXAS" / "Ti_VASP" / "runs",
        "hidden_dims": UNIVERSAL_DIMS,
        "split": ("Ti", "VASP"),
        "batch_size": 64,
    },
]


def load_split(element: str, spectrum_type: str) -> MLSplits:
    data = {}
    for split_name in ["train", "val", "test"]:
        X = np.loadtxt(DATA_DIR / f"{element}_{spectrum_type}_{split_name}_X.txt", dtype=np.float32)
        y = np.loadtxt(DATA_DIR / f"{element}_{spectrum_type}_{split_name}_y.txt", dtype=np.float32)
        data[split_name] = MLData(X=X, y=y)
    return MLSplits(**data)


def checkpoint_paths(root: Path):
    """All best-model checkpoints under a model's output folder.

    Your current XASBlockRegressor runs use:
        runs/paper_.../best-model-epoch=...ckpt

    If you later train with a raw Hydra Trainer, it may use:
        runs/paper_.../checkpoints/best-model-epoch=...ckpt

    Supporting both does not change your current results; it just avoids missing
    future checkpoints.
    """
    root = Path(root)
    paths = []
    paths += list(root.glob("*/best*.ckpt"))
    paths += list(root.glob("*/checkpoints/best*.ckpt"))
    return sorted(set(paths))


def checkpoint_seed(path: Path):
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else np.nan


def run_dir_from_checkpoint(path: Path) -> Path:
    path = Path(path)
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def median_mse(y_true, y_pred) -> float:
    return float(np.median(np.mean((y_true - y_pred) ** 2, axis=1)))


def predict_without_lightning_trainer(model: XASBlockRegressor, X: np.ndarray, batch_size: int = 1024):
    """Single-process prediction.

    XASBlockRegressor.predict() creates a Lightning Trainer. On a machine with
    multiple visible GPUs, Lightning may auto-start DDP and return only this
    process's shard of predictions. Direct torch inference avoids that.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = model.model.to(device).eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(module(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_checkpoint(path: Path, hidden_dims, split: MLSplits, batch_size: int):
    model = XASBlockRegressor(
        directory=str(Path(path).parent),
        overwrite_save_dir=False,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=list(hidden_dims),
        batch_size=batch_size,
        max_epochs=1,
    )
    model.load("best")

    pred = predict_without_lightning_trainer(model, split.test.X)
    target = split.test.y
    metrics = ModelMetrics(predictions=pred, targets=target)

    baseline = np.repeat(split.train.y.mean(axis=0, keepdims=True), len(target), axis=0)
    baseline_median = median_mse(target, baseline)
    model_median = float(metrics.median_of_mse_per_spectra)

    return {
        "mse": float(metrics.mse),
        "median_mse": model_median,
        "baseline_median_mse": baseline_median,
        "eta": baseline_median / model_median,
    }


def main():
    parser = argparse.ArgumentParser(description="Find best Ti eta checkpoints in output/training.")
    parser.add_argument(
        "--delete-non-best",
        action="store_true",
        help="Delete non-winning run folders after evaluating. Default only reports/saves results.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    split_cache = {}
    best_rows = []
    audit_rows = []
    best_run_dirs = set()
    candidate_run_dirs = set()

    for spec in MODEL_SPECS:
        split_key = spec["split"]
        if split_key not in split_cache:
            split_cache[split_key] = load_split(*split_key)
        split = split_cache[split_key]

        paths = checkpoint_paths(spec["root"])
        if not paths:
            raise FileNotFoundError(f"No best checkpoints found under {spec['root']}")

        candidates = []
        for path in paths:
            scores = evaluate_checkpoint(path, spec["hidden_dims"], split, spec["batch_size"])
            row = {
                "checkpoint_seed": checkpoint_seed(path),
                "dataset": spec["dataset"],
                "model": spec["model"],
                **scores,
                "paper_eta": PAPER_ETA[(spec["dataset"], spec["model"])],
                "checkpoint_path": str(path),
                "run_dir": str(run_dir_from_checkpoint(path)),
            }
            candidates.append(row)
            audit_rows.append(row)

        candidates.sort(key=lambda r: (-r["eta"], r["median_mse"], r["checkpoint_path"]))
        best = candidates[0]
        best_run_dirs.add(best["run_dir"])
        candidate_run_dirs.update(row["run_dir"] for row in candidates)
        best_rows.append({k: v for k, v in best.items() if k not in {"checkpoint_path", "run_dir"}})

        print(f"{spec['dataset']} / {spec['model']}: eta={best['eta']:.6f}, median_mse={best['median_mse']:.6g}")
        print(f"  {best['checkpoint_path']}")

    best_df = pd.DataFrame(best_rows)
    audit_df = pd.DataFrame(audit_rows).sort_values(["dataset", "model", "eta"], ascending=[True, True, False])

    best_csv = RESULTS_DIR / "best_eta_ti_results.csv"
    audit_csv = RESULTS_DIR / "best_eta_ti_all_candidates.csv"
    best_df.to_csv(best_csv, index=False)
    audit_df.to_csv(audit_csv, index=False)

    print("\nBest eta table:")
    print(best_df.to_string(index=False))
    print("\nSaved:", best_csv)
    print("Saved all candidates:", audit_csv)

    prune_dirs = sorted(Path(p) for p in (candidate_run_dirs - best_run_dirs))
    if prune_dirs:
        print("\nNon-winning run folders:")
        for path in prune_dirs:
            print("  DELETE" if args.delete_non_best else "  would delete", path)
        if args.delete_non_best:
            for path in prune_dirs:
                shutil.rmtree(path)
            print(f"Deleted {len(prune_dirs)} non-winning run folders.")
        else:
            print("\nTo delete these folders, rerun with:")
            print("  python tutorial_omnixas/find_best_eta_ti.py --delete-non-best")
    else:
        print("\nNo non-winning run folders found.")

    return best_df


if __name__ == "__main__":
    main()
