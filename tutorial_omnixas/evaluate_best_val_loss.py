#!/usr/bin/env python3
"""Select best-validation-loss checkpoints and report test eta.

Default scans all FEFF elements (Ti-Cu) plus available Ti/Cu VASP runs.
Selection uses exact validation loss stored inside Lightning checkpoints, not the
rounded val_loss in checkpoint filenames and not test eta.
"""

from datetime import datetime
from pathlib import Path
import argparse
import re
import shutil

parser = argparse.ArgumentParser(description="Select best-val-loss checkpoints and report test eta.")
parser.add_argument("--elements", nargs="+", default=["all"], help="Elements to scan, e.g. Ti Cu or all")
parser.add_argument("--no-vasp", action="store_true", help="Only scan FEFF datasets")
parser.add_argument("--delete-non-best", action="store_true", help="Delete non-winning run folders after evaluating")
args = parser.parse_args()

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
RESULTS_DIR = OUTPUT_ROOT / "comparisons" / "best_val_loss" / datetime.now().strftime("%Y%m%d_%H%M%S")

INPUT_DIM, OUTPUT_DIM = 64, 141
FEFF_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
UNIVERSAL_DIMS = [500, 500, 550]
FEFF_HPARAMS = {
    "Ti": {"batch_size": 32, "widths": [600, 600, 450]},
    "V":  {"batch_size": 32, "widths": [600, 550, 450]},
    "Cr": {"batch_size": 32, "widths": [450, 350, 150]},
    "Mn": {"batch_size": 64, "widths": [500, 400, 300]},
    "Fe": {"batch_size": 64, "widths": [450, 400, 450]},
    "Co": {"batch_size": 32, "widths": [600, 550, 450]},
    "Ni": {"batch_size": 32, "widths": [600, 300]},
    "Cu": {"batch_size": 32, "widths": [600, 600, 400]},
}
VASP_HPARAMS = {
    "Ti": {"batch_size": 64, "widths": [500, 600, 400]},
    "Cu": {"batch_size": 64, "widths": [550, 600, 450]},
}
PAPER_ETA = {
    ("Ti", "FEFF", "ExpertXAS"): 6.35,
    ("Ti", "FEFF", "UniversalXAS"): 4.19,
    ("Ti", "FEFF", "Tuned-UniversalXAS"): 7.63,
    ("V", "FEFF", "ExpertXAS"): 7.30,
    ("V", "FEFF", "UniversalXAS"): 5.19,
    ("V", "FEFF", "Tuned-UniversalXAS"): 9.22,
    ("Cr", "FEFF", "ExpertXAS"): 8.54,
    ("Cr", "FEFF", "UniversalXAS"): 7.13,
    ("Cr", "FEFF", "Tuned-UniversalXAS"): 10.44,
    ("Mn", "FEFF", "ExpertXAS"): 17.66,
    ("Mn", "FEFF", "UniversalXAS"): 13.15,
    ("Mn", "FEFF", "Tuned-UniversalXAS"): 29.81,
    ("Fe", "FEFF", "ExpertXAS"): 7.51,
    ("Fe", "FEFF", "UniversalXAS"): 6.04,
    ("Fe", "FEFF", "Tuned-UniversalXAS"): 8.98,
    ("Co", "FEFF", "ExpertXAS"): 14.47,
    ("Co", "FEFF", "UniversalXAS"): 9.58,
    ("Co", "FEFF", "Tuned-UniversalXAS"): 19.83,
    ("Ni", "FEFF", "ExpertXAS"): 8.45,
    ("Ni", "FEFF", "UniversalXAS"): 6.43,
    ("Ni", "FEFF", "Tuned-UniversalXAS"): 11.21,
    ("Cu", "FEFF", "ExpertXAS"): 5.19,
    ("Cu", "FEFF", "UniversalXAS"): 2.75,
    ("Cu", "FEFF", "Tuned-UniversalXAS"): 4.81,
    ("Ti", "VASP", "ExpertXAS"): 4.75,
    ("Ti", "VASP", "Tuned-UniversalXAS"): 5.27,
    ("Cu", "VASP", "ExpertXAS"): 8.46,
    ("Cu", "VASP", "Tuned-UniversalXAS"): 9.21,
}


def split_exists(element, typ):
    return (DATA_DIR / f"{element}_{typ}_test_X.txt").exists()


def load_split(element, typ):
    return MLSplits(**{
        name: MLData(
            X=np.loadtxt(DATA_DIR / f"{element}_{typ}_{name}_X.txt", dtype=np.float32),
            y=np.loadtxt(DATA_DIR / f"{element}_{typ}_{name}_y.txt", dtype=np.float32),
        )
        for name in ["train", "val", "test"]
    })


def run_dir(ckpt_path):
    ckpt_path = Path(ckpt_path)
    return ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent


def checkpoint_val_loss(ckpt_path):
    try:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        scores = []
        for callback_name, state in ckpt.get("callbacks", {}).items():
            if "ModelCheckpoint" not in str(callback_name):
                continue
            score = state.get("best_model_score")
            if score is not None:
                scores.append(float(score.detach().cpu().item() if torch.is_tensor(score) else score))
        if scores:
            return min(scores)
    except Exception as exc:
        print(f"Warning: could not read exact val_loss from {ckpt_path}: {exc}")

    match = re.search(r"val_loss[=_](\d+(?:\.\d+)?)", Path(ckpt_path).name)
    return float(match.group(1)) if match else float("inf")


def main():
    elements = FEFF_ELEMENTS if "all" in args.elements else args.elements
    specs = []
    for element in elements:
        if split_exists(element, "FEFF"):
            h = FEFF_HPARAMS[element]
            specs.extend([
                (element, "FEFF", "ExpertXAS", OUTPUT_ROOT / "expertXAS" / f"{element}_FEFF" / "runs", h["widths"], h["batch_size"]),
                (element, "FEFF", "UniversalXAS", OUTPUT_ROOT / "universalXAS" / "All_FEFF" / "runs", UNIVERSAL_DIMS, 32),
                (element, "FEFF", "Tuned-UniversalXAS", OUTPUT_ROOT / "tunedUniversalXAS" / f"{element}_FEFF" / "runs", UNIVERSAL_DIMS, h["batch_size"]),
            ])
        if not args.no_vasp and element in VASP_HPARAMS and split_exists(element, "VASP"):
            h = VASP_HPARAMS[element]
            specs.extend([
                (element, "VASP", "ExpertXAS", OUTPUT_ROOT / "expertXAS" / f"{element}_VASP" / "runs", h["widths"], h["batch_size"]),
                (element, "VASP", "Tuned-UniversalXAS", OUTPUT_ROOT / "tunedUniversalXAS" / f"{element}_VASP" / "runs", UNIVERSAL_DIMS, h["batch_size"]),
            ])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    best_rows, audit_rows = [], []
    best_run_dirs, candidate_run_dirs = set(), set()
    split_cache = {}

    for element, typ, model_name, root, dims, batch_size in specs:
        ckpts = sorted(set(root.glob("*/best*.ckpt")) | set(root.glob("*/checkpoints/best*.ckpt")))
        if not ckpts:
            print(f"Skipping {element} {typ} / {model_name}: no checkpoints in {root}")
            continue

        split_cache.setdefault((element, typ), load_split(element, typ))
        split = split_cache[(element, typ)]
        target = split.test.y
        baseline = np.repeat(split.train.y.mean(axis=0, keepdims=True), len(target), axis=0)
        baseline_median = float(np.median(np.mean((target - baseline) ** 2, axis=1)))

        candidates = []
        for ckpt in ckpts:
            model = XASBlockRegressor(
                directory=str(run_dir(ckpt)),
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

            metrics = ModelMetrics(predictions=pred, targets=target)
            median_mse = float(metrics.median_of_mse_per_spectra)
            val_loss = checkpoint_val_loss(ckpt)
            row = {
                "checkpoint_seed": int(re.search(r"seed(\d+)", str(ckpt)).group(1)) if re.search(r"seed(\d+)", str(ckpt)) else np.nan,
                "element": element,
                "type": typ,
                "dataset": f"{element} {typ}",
                "model": model_name,
                "mse": float(metrics.mse),
                "median_mse": median_mse,
                "baseline_median_mse": baseline_median,
                "eta": baseline_median / median_mse,
                "val_loss": val_loss,
                "paper_eta": PAPER_ETA.get((element, typ, model_name), np.nan),
                "checkpoint_path": str(ckpt),
                "run_dir": str(run_dir(ckpt)),
            }
            candidates.append(row)
            audit_rows.append(row)

        candidates.sort(key=lambda row: (row["val_loss"], row["checkpoint_path"]))
        best = candidates[0]
        best_run_dirs.add(best["run_dir"])
        candidate_run_dirs.update(row["run_dir"] for row in candidates)
        best_rows.append({k: v for k, v in best.items() if k not in {"checkpoint_path", "run_dir"}})
        print(f"{best['dataset']} / {model_name}: val_loss={best['val_loss']:.8g}, test eta={best['eta']:.6f}")
        print(f"  {best['checkpoint_path']}")

    best_df = pd.DataFrame(best_rows)
    audit_df = pd.DataFrame(audit_rows).sort_values(["element", "type", "model", "val_loss"])
    best_csv = RESULTS_DIR / "best_val_loss_results.csv"
    audit_csv = RESULTS_DIR / "best_val_loss_all_candidates.csv"
    best_df.to_csv(best_csv, index=False)
    audit_df.to_csv(audit_csv, index=False)

    print("\nBest validation-loss-selected table:")
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
