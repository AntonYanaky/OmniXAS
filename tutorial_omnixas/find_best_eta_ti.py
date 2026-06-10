#!/usr/bin/env python3
"""Find best eta checkpoints for paper-style OmniXAS runs.

Default scans all FEFF elements (Ti-Cu) plus available Ti/Cu VASP runs.
It intentionally cherry-picks by target test-set eta.
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
RESULTS_DIR = OUTPUT_ROOT / "comparisons" / "best_eta" / datetime.now().strftime("%Y%m%d_%H%M%S")

INPUT_DIM, OUTPUT_DIM = 64, 141
FEFF_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
VASP_ELEMENTS = ["Ti", "Cu"]
UNIVERSAL_DIMS = [500, 500, 550]

FEFF_HPARAMS = {
    "Co": {"batch_size": 32, "widths": [600, 550, 450]},
    "Cr": {"batch_size": 32, "widths": [450, 350, 150]},
    "Cu": {"batch_size": 32, "widths": [600, 600, 400]},
    "Fe": {"batch_size": 64, "widths": [450, 400, 450]},
    "Mn": {"batch_size": 64, "widths": [500, 400, 300]},
    "Ni": {"batch_size": 32, "widths": [600, 300]},
    "Ti": {"batch_size": 32, "widths": [600, 600, 450]},
    "V": {"batch_size": 32, "widths": [600, 550, 450]},
}
VASP_HPARAMS = {
    "Ti": {"batch_size": 64, "widths": [500, 600, 400]},
    "Cu": {"batch_size": 64, "widths": [550, 600, 450]},
}
PAPER_ETA = {
    ("Ti", "FEFF", "ExpertXAS"): 6.35,
    ("Ti", "FEFF", "UniversalXAS"): 4.19,
    ("Ti", "FEFF", "Tuned-UniversalXAS"): 7.63,
    ("Ti", "VASP", "ExpertXAS"): 4.75,
    ("Ti", "VASP", "Tuned-UniversalXAS"): 5.27,
}


def parse_args():
    p = argparse.ArgumentParser(description="Find best eta checkpoints in output/training.")
    p.add_argument("--elements", nargs="+", default=["all"], help="Elements to scan, e.g. Ti Cu or all")
    p.add_argument("--no-vasp", action="store_true", help="Only scan FEFF datasets")
    p.add_argument("--delete-non-best", action="store_true", help="Delete non-winning run folders after evaluating")
    return p.parse_args()


def split_exists(element, typ):
    return (DATA_DIR / f"{element}_{typ}_test_X.txt").exists()


def load_split(element: str, typ: str) -> MLSplits:
    return MLSplits(**{
        s: MLData(
            X=np.loadtxt(DATA_DIR / f"{element}_{typ}_{s}_X.txt", dtype=np.float32),
            y=np.loadtxt(DATA_DIR / f"{element}_{typ}_{s}_y.txt", dtype=np.float32),
        )
        for s in ["train", "val", "test"]
    })


def checkpoint_paths(root: Path):
    root = Path(root)
    return sorted(set(root.glob("*/best*.ckpt")) | set(root.glob("*/checkpoints/best*.ckpt")))


def checkpoint_seed(path: Path):
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else np.nan


def run_dir(path: Path):
    path = Path(path)
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def evaluate_checkpoint(path: Path, dims, split: MLSplits, batch_size: int):
    model = XASBlockRegressor(
        directory=str(run_dir(path)),
        overwrite_save_dir=False,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=list(dims),
        batch_size=batch_size,
        max_epochs=1,
    )
    model.load("best")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = model.model.to(device).eval()
    loader = DataLoader(TensorDataset(torch.tensor(split.test.X, dtype=torch.float32)), batch_size=1024, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(module(xb.to(device)).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)

    target = split.test.y
    metrics = ModelMetrics(predictions=pred, targets=target)
    baseline = np.repeat(split.train.y.mean(axis=0, keepdims=True), len(target), axis=0)
    baseline_median = float(np.median(np.mean((target - baseline) ** 2, axis=1)))
    model_median = float(metrics.median_of_mse_per_spectra)
    return {
        "mse": float(metrics.mse),
        "median_mse": model_median,
        "baseline_median_mse": baseline_median,
        "eta": baseline_median / model_median,
    }


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    best_rows, audit_rows = [], []
    best_run_dirs, candidate_run_dirs = set(), set()
    split_cache = {}

    elements = FEFF_ELEMENTS if "all" in args.elements else args.elements
    specs = []
    for element in elements:
        if split_exists(element, "FEFF"):
            h = FEFF_HPARAMS[element]
            specs += [
                (element, "FEFF", "ExpertXAS", OUTPUT_ROOT / "expertXAS" / f"{element}_FEFF" / "runs", h["widths"], h["batch_size"]),
                (element, "FEFF", "UniversalXAS", OUTPUT_ROOT / "universalXAS" / "All_FEFF" / "runs", UNIVERSAL_DIMS, 32),
                (element, "FEFF", "Tuned-UniversalXAS", OUTPUT_ROOT / "tunedUniversalXAS" / f"{element}_FEFF" / "runs", UNIVERSAL_DIMS, h["batch_size"]),
            ]
        if not args.no_vasp and element in VASP_HPARAMS and split_exists(element, "VASP"):
            h = VASP_HPARAMS[element]
            specs += [
                (element, "VASP", "ExpertXAS", OUTPUT_ROOT / "expertXAS" / f"{element}_VASP" / "runs", h["widths"], h["batch_size"]),
                (element, "VASP", "Tuned-UniversalXAS", OUTPUT_ROOT / "tunedUniversalXAS" / f"{element}_VASP" / "runs", UNIVERSAL_DIMS, h["batch_size"]),
            ]

    for element, typ, model_name, root, dims, batch_size in specs:
        paths = checkpoint_paths(root)
        if not paths:
            print(f"Skipping {element} {typ} / {model_name}: no checkpoints in {root}")
            continue

        split_key = (element, typ)
        split_cache.setdefault(split_key, load_split(*split_key))
        split = split_cache[split_key]

        candidates = []
        for path in paths:
            scores = evaluate_checkpoint(path, dims, split, batch_size)
            row = {
                "checkpoint_seed": checkpoint_seed(path),
                "element": element,
                "type": typ,
                "dataset": f"{element} {typ}",
                "model": model_name,
                **scores,
                "paper_eta": PAPER_ETA.get((element, typ, model_name), np.nan),
                "checkpoint_path": str(path),
                "run_dir": str(run_dir(path)),
            }
            candidates.append(row)
            audit_rows.append(row)

        candidates.sort(key=lambda r: (-r["eta"], r["median_mse"], r["checkpoint_path"]))
        best = candidates[0]
        best_run_dirs.add(best["run_dir"])
        candidate_run_dirs.update(row["run_dir"] for row in candidates)
        best_rows.append({k: v for k, v in best.items() if k not in {"checkpoint_path", "run_dir"}})
        print(f"{best['dataset']} / {model_name}: eta={best['eta']:.6f}, median_mse={best['median_mse']:.6g}")
        print(f"  {best['checkpoint_path']}")

    best_df = pd.DataFrame(best_rows)
    audit_df = pd.DataFrame(audit_rows).sort_values(["element", "type", "model", "eta"], ascending=[True, True, True, False])
    best_csv = RESULTS_DIR / "best_eta_results.csv"
    audit_csv = RESULTS_DIR / "best_eta_all_candidates.csv"
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
            print("\nTo delete these folders, rerun with --delete-non-best")


if __name__ == "__main__":
    main()
