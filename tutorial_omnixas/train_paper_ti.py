#!/usr/bin/env python3
"""Train paper-style Ti OmniXAS models. Saves checkpoints only."""

import argparse, os, random, re
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", required=True,
               choices=["all", "experts", "tuned", "universal_feff", "ti_feff_expert", "ti_vasp_expert", "ti_feff_tuned", "ti_vasp_tuned"])
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
UNIVERSAL_DIMS = [500, 500, 550]
TI_FEFF_DIMS = [600, 600, 450]
TI_VASP_DIMS = [500, 600, 400]
DEFAULT_DROPOUT = 0.5
TUNED_DROPOUTS = [0.5, 0.0]
MAX_EPOCHS = 1000
PATIENCE = 25
INITIAL_LR = 1e-2
MIN_LR = 1e-4

RUNS = {
    "universal_feff": OUT / "universalXAS" / "All_FEFF" / "runs",
    "ti_feff_expert": OUT / "expertXAS" / "Ti_FEFF" / "runs",
    "ti_vasp_expert": OUT / "expertXAS" / "Ti_VASP" / "runs",
    "ti_feff_tuned": OUT / "tunedUniversalXAS" / "Ti_FEFF" / "runs",
    "ti_vasp_tuned": OUT / "tunedUniversalXAS" / "Ti_VASP" / "runs",
}


def vloss(path):
    m = re.search(r"val_loss[=_](\d+(?:\.\d+)?)", Path(path).name)
    return float(m.group(1)) if m else float("inf")


def best(runs, label):
    ckpts = sorted(Path(runs).glob("paper_*/best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No paper_* checkpoint for {label}. Train UniversalXAS first if fine-tuning.")
    ckpt = min(ckpts, key=vloss)
    print(f"Best {label}: {ckpt}", flush=True)
    return ckpt


def split(element, typ):
    d = {}
    for s in ["train", "val", "test"]:
        X = np.loadtxt(DATA / f"{element}_{typ}_{s}_X.txt", dtype=np.float32)
        y = np.loadtxt(DATA / f"{element}_{typ}_{s}_y.txt", dtype=np.float32)
        assert X.shape[1] == INPUT_DIM and y.shape[1] == OUTPUT_DIM
        d[s] = MLData(X=X, y=y)
    return MLSplits(**d)


def save_dir(runs, seed, dropout=None):
    name = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seed{seed}"
    if dropout is not None:
        name += f"_dropout{str(dropout).replace('.', 'p')}"
    path = Path(runs) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def reg(directory, dims, batch):
    return XASBlockRegressor(
        directory=str(directory), overwrite_save_dir=False,
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dims=dims, batch_size=batch,
        max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
        initial_lr=INITIAL_LR, min_lr=MIN_LR,
    )


def banner(i, total, text):
    print("\n" + "=" * 90, flush=True)
    print(f"JOB {i}/{total}: {text}", flush=True)
    print("=" * 90, flush=True)


aliases = {
    "all": ["universal_feff", "ti_feff_expert", "ti_vasp_expert", "ti_feff_tuned", "ti_vasp_tuned"],
    "experts": ["ti_feff_expert", "ti_vasp_expert"],
    "tuned": ["ti_feff_tuned", "ti_vasp_tuned"],
}
selected = []
for m in args.models:
    selected += aliases.get(m, [m])
selected = [m for m in ["universal_feff", "ti_feff_expert", "ti_vasp_expert", "ti_feff_tuned", "ti_vasp_tuned"] if m in set(selected)]
seeds = ([args.seed] if args.seed is not None and args.n_runs == 1
         else [(random.Random(args.seed) if args.seed is not None else random.SystemRandom()).randint(0, 2**32 - 1) for _ in range(args.n_runs)])
total = sum(args.n_runs for m in selected if m in {"universal_feff", "ti_feff_expert", "ti_vasp_expert"})
total += sum(args.n_runs * len(TUNED_DROPOUTS) for m in selected if m in {"ti_feff_tuned", "ti_vasp_tuned"})
job = 0

torch.set_float32_matmul_precision("high")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available", flush=True)
print("Selected:", selected, flush=True)
print("Seeds:", seeds, flush=True)
print("Jobs:", total, flush=True)
print("Folder format: paper_<timestamp>_seed<seed>[_dropout0p5]", flush=True)

assert DATA.exists(), f"Missing data directory: {DATA}"
els = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
feff = {e: split(e, "FEFF") for e in els}
ti_feff, ti_vasp = feff["Ti"], split("Ti", "VASP")
univ = MLSplits(
    train=MLData(X=np.concatenate([feff[e].train.X for e in els]), y=np.concatenate([feff[e].train.y for e in els])),
    val=MLData(X=np.concatenate([feff[e].val.X for e in els]), y=np.concatenate([feff[e].val.y for e in els])),
    test=MLData(X=np.concatenate([feff[e].test.X for e in els]), y=np.concatenate([feff[e].test.y for e in els])),
)
print("Universal FEFF:", univ.train.X.shape, univ.val.X.shape, univ.test.X.shape, flush=True)
print("Ti FEFF:", ti_feff.train.X.shape, ti_feff.val.X.shape, ti_feff.test.X.shape, flush=True)
print("Ti VASP:", ti_vasp.train.X.shape, ti_vasp.val.X.shape, ti_vasp.test.X.shape, flush=True)

scratch = {
    "universal_feff": ("UniversalXAS FEFF", RUNS["universal_feff"], univ, UNIVERSAL_DIMS, 32),
    "ti_feff_expert": ("Ti FEFF ExpertXAS", RUNS["ti_feff_expert"], ti_feff, TI_FEFF_DIMS, 32),
    "ti_vasp_expert": ("Ti VASP ExpertXAS", RUNS["ti_vasp_expert"], ti_vasp, TI_VASP_DIMS, 64),
}
for key, (label, runs, data, dims, batch) in scratch.items():
    if key not in selected:
        continue
    XASBlock.DROPOUT = DEFAULT_DROPOUT
    ckpts = []
    for seed in seeds:
        job += 1
        seed_everything(seed, workers=True)
        d = save_dir(runs, seed)
        banner(job, total, f"training {label} | seed={seed} | dir={d}")
        model = reg(d, dims, batch)
        model.fit(data)  # Lightning prints epoch/progress output.
        ckpts += sorted(d.glob("best*.ckpt"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Best new {label}: {min(ckpts, key=vloss)}", flush=True)

tuned = {
    "ti_feff_tuned": ("Ti FEFF Tuned-UniversalXAS", RUNS["ti_feff_tuned"], ti_feff, 32),
    "ti_vasp_tuned": ("Ti VASP Tuned-UniversalXAS", RUNS["ti_vasp_tuned"], ti_vasp, 64),
}
for key, (label, runs, data, batch) in tuned.items():
    if key not in selected:
        continue
    source = best(RUNS["universal_feff"], "UniversalXAS source").parent
    ckpts = []
    for seed in seeds:
        for dropout in TUNED_DROPOUTS:
            job += 1
            seed_everything(seed, workers=True)
            XASBlock.DROPOUT = dropout
            d = save_dir(runs, seed, dropout)
            banner(job, total, f"fine-tuning {label} | seed={seed} | dropout={dropout} | dir={d}")
            model = reg(source, UNIVERSAL_DIMS, batch)
            model.load("best")
            model.cfg.directory = str(d)
            model.fit(data)  # Lightning prints epoch/progress output.
            ckpts += sorted(d.glob("best*.ckpt"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Best new {label}: {min(ckpts, key=vloss)}", flush=True)

print("\nDone. Checkpoints saved under output/training/.../runs/paper_*", flush=True)
