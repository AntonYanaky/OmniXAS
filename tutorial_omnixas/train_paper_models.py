#!/usr/bin/env python3
"""Headless paper-style OmniXAS training.

Model family is selected with --models. Elements are selected with --elements.
Spectrum type is selected with --types because FEFF and VASP are separate datasets.

Examples:
  # Universal foundation model; always all FEFF elements.
  python tutorial_omnixas/train_paper_models.py --models universal --seed 42 --gpu 0

  # FEFF experts and tuned models for all eight FEFF elements.
  python tutorial_omnixas/train_paper_models.py --models expert tuned --elements all --types FEFF --seed 42 --gpu 0

  # Ti/Cu VASP experts and tuned models.
  python tutorial_omnixas/train_paper_models.py --models expert tuned --elements Ti Cu --types VASP --seed 42 --gpu 0

  # Everything available.
  python tutorial_omnixas/train_paper_models.py --models all --elements all --types all --seed 42 --gpu 0
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", required=True, choices=["all", "universal", "expert", "tuned"])
p.add_argument("--elements", nargs="+", default=["all"], help="Elements for expert/tuned models: all, Ti, V, Cr, Mn, Fe, Co, Ni, Cu")
p.add_argument("--types", nargs="+", default=["FEFF"], choices=["all", "FEFF", "VASP"], help="Spectrum types for expert/tuned models. Universal is always FEFF.")
p.add_argument("--n-runs", type=int, default=1)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--gpu", type=str, default=None)
args = p.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.n_runs < 1:
    raise ValueError("--n-runs must be >= 1")

import numpy as np
import torch
from lightning.pytorch import seed_everything

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.xasblock import XASBlock
from omnixas.model.xasblock_regressor import XASBlockRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tutorial_omnixas" / "ml_data"
OUT = ROOT / "output" / "training"

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
DEFAULT_DROPOUT = 0.5
TUNED_DROPOUTS = [0.5, 0.0]
MAX_EPOCHS = 1000
PATIENCE = 25
INITIAL_LR = 1e-2
MIN_LR = 1e-4


def split_exists(element, typ):
    return (DATA / f"{element}_{typ}_train_X.txt").exists()


def split(element, typ):
    return MLSplits(**{
        s: MLData(
            X=np.loadtxt(DATA / f"{element}_{typ}_{s}_X.txt", dtype=np.float32),
            y=np.loadtxt(DATA / f"{element}_{typ}_{s}_y.txt", dtype=np.float32),
        )
        for s in ["train", "val", "test"]
    })


def run_root(kind, element=None, typ=None):
    if kind == "universal":
        return OUT / "universalXAS" / "All_FEFF" / "runs"
    folder = "expertXAS" if kind == "expert" else "tunedUniversalXAS"
    return OUT / folder / f"{element}_{typ}" / "runs"


def save_dir(root, seed, dropout=None):
    name = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seed{seed}"
    if dropout is not None:
        name += f"_dropout{str(dropout).replace('.', 'p')}"
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def reg(directory, dims, batch):
    return XASBlockRegressor(
        directory=str(directory),
        overwrite_save_dir=False,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=list(dims),
        batch_size=batch,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=PATIENCE,
        initial_lr=INITIAL_LR,
        min_lr=MIN_LR,
    )


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
        print(f"Warning: could not read exact val_loss from {ckpt_path}: {exc}", flush=True)

    import re
    match = re.search(r"val_loss[=_](\d+(?:\.\d+)?)", Path(ckpt_path).name)
    return float(match.group(1)) if match else float("inf")


def best_universal_source_by_val_loss(label):
    ckpts = sorted(run_root("universal").glob("paper_*/best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No UniversalXAS checkpoints found. Train UniversalXAS first.")
    best_ckpt = min(ckpts, key=checkpoint_val_loss)
    print(f"Best UniversalXAS source for {label}: val_loss={checkpoint_val_loss(best_ckpt):.8g} | {best_ckpt}", flush=True)
    return best_ckpt.parent


def banner(i, total, text):
    print("\n" + "=" * 90, flush=True)
    prefix = f"JOB {i}/{total}" if total else f"JOB {i}"
    print(f"{prefix}: {text}", flush=True)
    print("=" * 90, flush=True)


models = ["universal", "expert", "tuned"] if "all" in args.models else list(dict.fromkeys(args.models))
elements = FEFF_ELEMENTS if "all" in args.elements else args.elements
bad_elements = [e for e in elements if e not in FEFF_ELEMENTS]
if bad_elements:
    raise ValueError(f"Unknown elements: {bad_elements}. Use one of {FEFF_ELEMENTS} or all.")
types = ["FEFF", "VASP"] if "all" in args.types else list(dict.fromkeys(args.types))
seeds = ([args.seed] if args.seed is not None and args.n_runs == 1
         else [(random.Random(args.seed) if args.seed is not None else random.SystemRandom()).randint(0, 2**32 - 1) for _ in range(args.n_runs)])

assert DATA.exists(), f"Missing data directory: {DATA}"
torch.set_float32_matmul_precision("high")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available", flush=True)
print("Models:", models, flush=True)
print("Elements:", elements, flush=True)
print("Types:", types, flush=True)
print("Seeds:", seeds, flush=True)

feff_splits = {e: split(e, "FEFF") for e in FEFF_ELEMENTS if split_exists(e, "FEFF")}
universal_parts = [feff_splits[e] for e in FEFF_ELEMENTS]
universal_split = MLSplits(
    train=MLData(X=np.concatenate([s.train.X for s in universal_parts]), y=np.concatenate([s.train.y for s in universal_parts])),
    val=MLData(X=np.concatenate([s.val.X for s in universal_parts]), y=np.concatenate([s.val.y for s in universal_parts])),
    test=MLData(X=np.concatenate([s.test.X for s in universal_parts]), y=np.concatenate([s.test.y for s in universal_parts])),
)
job = 0

if "universal" in models:
    for seed in seeds:
        job += 1
        seed_everything(seed, workers=True)
        XASBlock.DROPOUT = DEFAULT_DROPOUT
        d = save_dir(run_root("universal"), seed)
        banner(job, 0, f"training UniversalXAS FEFF | seed={seed} | dir={d}")
        reg(d, UNIVERSAL_DIMS, 32).fit(universal_split)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

for element in elements:
    for typ in types:
        if not split_exists(element, typ):
            print(f"Skipping {element} {typ}: no data")
            continue
        if typ == "FEFF":
            hparams = FEFF_HPARAMS[element]
            data = feff_splits[element]
        elif typ == "VASP":
            if element not in VASP_HPARAMS:
                print(f"Skipping {element} VASP: no VASP hparams/data expected")
                continue
            hparams = VASP_HPARAMS[element]
            data = split(element, "VASP")
        else:
            raise ValueError(typ)

        if "expert" in models:
            for seed in seeds:
                job += 1
                seed_everything(seed, workers=True)
                XASBlock.DROPOUT = DEFAULT_DROPOUT
                d = save_dir(run_root("expert", element, typ), seed)
                banner(job, 0, f"training {element} {typ} ExpertXAS | seed={seed} | dir={d}")
                reg(d, hparams["widths"], hparams["batch_size"]).fit(data)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if "tuned" in models:
            source = best_universal_source_by_val_loss(f"{element} {typ} Tuned-UniversalXAS")
            for seed in seeds:
                for dropout in TUNED_DROPOUTS:
                    job += 1
                    seed_everything(seed, workers=True)
                    XASBlock.DROPOUT = dropout
                    d = save_dir(run_root("tuned", element, typ), seed, dropout)
                    banner(job, 0, f"fine-tuning {element} {typ} Tuned-UniversalXAS | seed={seed} | dropout={dropout} | dir={d}")
                    model = reg(source, UNIVERSAL_DIMS, hparams["batch_size"])
                    model.load("best")
                    model.cfg.directory = str(d)
                    model.fit(data)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

print("\nDone. Run: python tutorial_omnixas/evaluate_best_val_loss.py", flush=True)
