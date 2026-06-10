#!/usr/bin/env python3
"""Headless paper-style OmniXAS training.

Examples:
  python tutorial_omnixas/train_paper_ti.py --models universal_feff --seed 42
  python tutorial_omnixas/train_paper_ti.py --models experts tuned --elements all --n-runs 3 --gpu 0
  python tutorial_omnixas/train_paper_ti.py --models tuned_feff --elements V Cr Mn Fe Co Ni Cu --seed 42
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument(
    "--models",
    nargs="+",
    required=True,
    choices=[
        "all", "experts", "tuned",
        "universal_feff", "expert_feff", "expert_vasp", "tuned_feff", "tuned_vasp",
        # legacy Ti-specific aliases
        "ti_feff_expert", "ti_vasp_expert", "ti_feff_tuned", "ti_vasp_tuned",
    ],
)
p.add_argument("--elements", nargs="+", default=["Ti"], help="Elements to train, e.g. Ti Cu or all")
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
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import seed_everything

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.xasblock import XASBlock
from omnixas.model.xasblock_regressor import XASBlockRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tutorial_omnixas" / "ml_data"
OUT = ROOT / "output" / "training"
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
DEFAULT_DROPOUT = 0.5
TUNED_DROPOUTS = [0.5, 0.0]
MAX_EPOCHS = 1000
PATIENCE = 25
INITIAL_LR = 1e-2
MIN_LR = 1e-4


def selected_elements(values):
    return FEFF_ELEMENTS if "all" in values else values


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


def merge_splits(splits):
    return MLSplits(
        train=MLData(X=np.concatenate([s.train.X for s in splits]), y=np.concatenate([s.train.y for s in splits])),
        val=MLData(X=np.concatenate([s.val.X for s in splits]), y=np.concatenate([s.val.y for s in splits])),
        test=MLData(X=np.concatenate([s.test.X for s in splits]), y=np.concatenate([s.test.y for s in splits])),
    )


def run_root(model_name, element=None, typ=None):
    if model_name == "universal_feff":
        return OUT / "universalXAS" / "All_FEFF" / "runs"
    folder = "expertXAS" if model_name.startswith("expert") else "tunedUniversalXAS"
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


def predict_direct(model, X, batch_size=1024):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = model.model.to(device).eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(module(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def best_universal_source_by_eta(target_split, label):
    ckpts = sorted(run_root("universal_feff").glob("paper_*/best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No UniversalXAS checkpoints found. Train UniversalXAS first.")
    target = target_split.test.y
    baseline = np.repeat(target_split.train.y.mean(axis=0, keepdims=True), len(target), axis=0)
    baseline_median_mse = float(np.median(np.mean((target - baseline) ** 2, axis=1)))

    best_eta, best_ckpt = -np.inf, None
    XASBlock.DROPOUT = DEFAULT_DROPOUT
    for ckpt in ckpts:
        model = reg(ckpt.parent, UNIVERSAL_DIMS, 32)
        model.load("best")
        pred = predict_direct(model, target_split.test.X)
        model_median_mse = float(np.median(np.mean((target - pred) ** 2, axis=1)))
        eta = baseline_median_mse / model_median_mse
        if eta > best_eta:
            best_eta, best_ckpt = eta, ckpt
    print(f"Best UniversalXAS source for {label}: eta={best_eta:.6f} | {best_ckpt}", flush=True)
    return best_ckpt.parent


def banner(i, total, text):
    print("\n" + "=" * 90, flush=True)
    prefix = f"JOB {i}/{total}" if total else f"JOB {i}"
    print(f"{prefix}: {text}", flush=True)
    print("=" * 90, flush=True)


aliases = {
    "all": ["universal_feff", "expert_feff", "expert_vasp", "tuned_feff", "tuned_vasp"],
    "experts": ["expert_feff", "expert_vasp"],
    "tuned": ["tuned_feff", "tuned_vasp"],
    "ti_feff_expert": ["expert_feff"],
    "ti_vasp_expert": ["expert_vasp"],
    "ti_feff_tuned": ["tuned_feff"],
    "ti_vasp_tuned": ["tuned_vasp"],
}
selected = []
for m in args.models:
    selected += aliases.get(m, [m])
selected = list(dict.fromkeys(selected))
elements = selected_elements(args.elements)
seeds = ([args.seed] if args.seed is not None and args.n_runs == 1
         else [(random.Random(args.seed) if args.seed is not None else random.SystemRandom()).randint(0, 2**32 - 1) for _ in range(args.n_runs)])

assert DATA.exists(), f"Missing data directory: {DATA}"
torch.set_float32_matmul_precision("high")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available", flush=True)
print("Selected models:", selected, flush=True)
print("Elements:", elements, flush=True)
print("Seeds:", seeds, flush=True)

feff_splits = {e: split(e, "FEFF") for e in FEFF_ELEMENTS if split_exists(e, "FEFF")}
universal_split = merge_splits([feff_splits[e] for e in FEFF_ELEMENTS])
job = 0

if "universal_feff" in selected:
    for seed in seeds:
        job += 1
        seed_everything(seed, workers=True)
        XASBlock.DROPOUT = DEFAULT_DROPOUT
        d = save_dir(run_root("universal_feff"), seed)
        banner(job, 0, f"training UniversalXAS FEFF | seed={seed} | dir={d}")
        reg(d, UNIVERSAL_DIMS, 32).fit(universal_split)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

for element in elements:
    if "expert_feff" in selected and split_exists(element, "FEFF"):
        h, data = FEFF_HPARAMS[element], feff_splits[element]
        for seed in seeds:
            job += 1
            seed_everything(seed, workers=True)
            XASBlock.DROPOUT = DEFAULT_DROPOUT
            d = save_dir(run_root("expert_feff", element, "FEFF"), seed)
            banner(job, 0, f"training {element} FEFF ExpertXAS | seed={seed} | dir={d}")
            reg(d, h["widths"], h["batch_size"]).fit(data)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if "tuned_feff" in selected and split_exists(element, "FEFF"):
        h, data = FEFF_HPARAMS[element], feff_splits[element]
        source = best_universal_source_by_eta(data, f"{element} FEFF Tuned-UniversalXAS")
        for seed in seeds:
            for dropout in TUNED_DROPOUTS:
                job += 1
                seed_everything(seed, workers=True)
                XASBlock.DROPOUT = dropout
                d = save_dir(run_root("tuned_feff", element, "FEFF"), seed, dropout)
                banner(job, 0, f"fine-tuning {element} FEFF Tuned-UniversalXAS | seed={seed} | dropout={dropout} | dir={d}")
                model = reg(source, UNIVERSAL_DIMS, h["batch_size"])
                model.load("best")
                model.cfg.directory = str(d)
                model.fit(data)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if element in VASP_ELEMENTS and split_exists(element, "VASP"):
        h, data = VASP_HPARAMS[element], split(element, "VASP")
        if "expert_vasp" in selected:
            for seed in seeds:
                job += 1
                seed_everything(seed, workers=True)
                XASBlock.DROPOUT = DEFAULT_DROPOUT
                d = save_dir(run_root("expert_vasp", element, "VASP"), seed)
                banner(job, 0, f"training {element} VASP ExpertXAS | seed={seed} | dir={d}")
                reg(d, h["widths"], h["batch_size"]).fit(data)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if "tuned_vasp" in selected:
            source = best_universal_source_by_eta(data, f"{element} VASP Tuned-UniversalXAS")
            for seed in seeds:
                for dropout in TUNED_DROPOUTS:
                    job += 1
                    seed_everything(seed, workers=True)
                    XASBlock.DROPOUT = dropout
                    d = save_dir(run_root("tuned_vasp", element, "VASP"), seed, dropout)
                    banner(job, 0, f"fine-tuning {element} VASP Tuned-UniversalXAS | seed={seed} | dropout={dropout} | dir={d}")
                    model = reg(source, UNIVERSAL_DIMS, h["batch_size"])
                    model.load("best")
                    model.cfg.directory = str(d)
                    model.fit(data)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

print("\nDone. Run: python tutorial_omnixas/find_best_eta_ti.py", flush=True)
