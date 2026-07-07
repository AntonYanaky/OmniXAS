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
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", required=True, choices=["all", "universal", "expert", "tuned"])
p.add_argument("--elements", nargs="+", default=["all"], help="Elements for expert/tuned models: all, Ti, V, Cr, Mn, Fe, Co, Ni, Cu")
p.add_argument("--types", nargs="+", default=["FEFF"], choices=["all", "FEFF", "VASP"], help="Spectrum types for expert/tuned models. Universal is always FEFF.")
p.add_argument("--n-runs", type=int, default=1)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--gpu", type=str, default=None)
p.add_argument("--data-dir", type=str, default="tutorial_omnixas/ml_data", help="Directory containing *_X.txt/*_y.txt split files.")
p.add_argument("--training-root", type=str, default="output/training", help="Root directory for model outputs.")
p.add_argument("--universal-lr", type=float, default=None, help="UniversalXAS LR. If omitted, keeps LR finder behavior.")
p.add_argument("--universal-dropout", type=float, default=None, help="UniversalXAS dropout. Defaults to paper value.")
p.add_argument("--universal-max-epochs", type=int, default=None)
p.add_argument("--universal-patience", type=int, default=None)
p.add_argument("--universal-monitor", choices=["val_loss", "val_median_mse"], default="val_median_mse")
p.add_argument("--universal-cos-lr", action="store_true", help="Use cosine LR for UniversalXAS.")
p.add_argument("--universal-cos-t", type=int, default=None)
p.add_argument("--universal-shuffle", action="store_true", help="Shuffle UniversalXAS training batches.")
p.add_argument(
    "--tuned-lr",
    type=float,
    default=None,
    help="Fine-tune LR for tuned models. Defaults to config/paper_hydra/train.yaml training.lr.",
)
p.add_argument("--tuned-dropouts", nargs="+", type=float, default=None, help="Dropout values for tuned models, e.g. --tuned-dropouts 0.0")
p.add_argument("--tuned-max-epochs", type=int, default=None, help="Max epochs for tuned models only.")
p.add_argument("--tuned-patience", type=int, default=None, help="Early-stopping patience for tuned models only.")
p.add_argument("--tuned-no-early-stopping", action="store_true", help="Disable early stopping for tuned models.")
p.add_argument("--tuned-batch-size", type=int, default=None, help="Override batch size for tuned models only.")
p.add_argument("--tuned-freeze-first-k", type=int, default=0, help="Freeze first k Linear layers during tuned fine-tuning.")
p.add_argument("--tuned-reset-final-layer", action="store_true", help="Reinitialize final Linear layer before tuned fine-tuning.")
p.add_argument("--tuned-reset-bn", action="store_true", help="Reset BatchNorm running stats before tuned fine-tuning.")
p.add_argument("--tuned-monitor", choices=["val_loss", "val_median_mse"], default="val_median_mse", help="Checkpoint/early-stop monitor for tuned models.")
p.add_argument("--tuned-cosine-lr", "--cos-lr", action="store_true", dest="tuned_cosine_lr", help="Use CosineAnnealingLR for tuned fine-tuning. Defaults: T_max=250 for FEFF, 600 for VASP; eta_min=1e-6.")
p.add_argument("--cos-t", type=int, default=None, help="Override tuned CosineAnnealingLR T_max.")
p.add_argument("--plateau-lr", action="store_true", help="Use ReduceLROnPlateau for tuned fine-tuning.")
p.add_argument("--warmup-cosine-lr", action="store_true", help="Use LinearLR warmup followed by CosineAnnealingLR for tuned fine-tuning.")
p.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs for --warmup-cosine-lr.")
p.add_argument("--warmup-start-factor", type=float, default=0.1, help="Starting LR factor for warmup, e.g. 0.1 starts at 10%% of --tuned-lr.")
p.add_argument("--onecycle-lr", action="store_true", help="Use OneCycleLR for tuned fine-tuning. --tuned-lr is used as max_lr.")
p.add_argument("--onecycle-pct-start", type=float, default=0.3, help="Fraction of OneCycle steps spent increasing LR.")
p.add_argument("--onecycle-div-factor", type=float, default=25.0, help="OneCycle initial LR divisor: initial_lr=max_lr/div_factor.")
p.add_argument("--onecycle-final-div-factor", type=float, default=1000.0, help="OneCycle final LR divisor after div_factor.")
p.add_argument("--tuned-source-val-eta", action="store_true", help="Select the UniversalXAS source by target validation eta instead of universal validation loss.")
args = p.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.n_runs < 1:
    raise ValueError("--n-runs must be >= 1")
selected_lr_schedulers = [
    args.tuned_cosine_lr,
    args.plateau_lr,
    args.warmup_cosine_lr,
    args.onecycle_lr,
]
if sum(bool(x) for x in selected_lr_schedulers) > 1:
    raise ValueError(
        "Use only one LR scheduler: --cos-lr, --plateau-lr, "
        "--warmup-cosine-lr, or --onecycle-lr"
    )

import numpy as np
import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.model.xasblock import XASBlock
from omnixas.model.xasblock_regressor import XASBlockRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(args.data_dir)
if not DATA.is_absolute():
    DATA = ROOT / DATA
OUT = Path(args.training_root)
if not OUT.is_absolute():
    OUT = ROOT / OUT
HYDRA_TRAIN_CFG = OmegaConf.load(ROOT / "config" / "paper_hydra" / "train.yaml")

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
TUNED_DROPOUTS = list(args.tuned_dropouts) if args.tuned_dropouts is not None else [0.5, 0.0]
MAX_EPOCHS = 1000
PATIENCE = 25
INITIAL_LR = 1e-3
UNIVERSAL_DROPOUT = DEFAULT_DROPOUT if args.universal_dropout is None else args.universal_dropout
UNIVERSAL_INITIAL_LR = INITIAL_LR if args.universal_lr is None else args.universal_lr
UNIVERSAL_USE_LR_FINDER = args.universal_lr is None
UNIVERSAL_MAX_EPOCHS = MAX_EPOCHS if args.universal_max_epochs is None else args.universal_max_epochs
UNIVERSAL_PATIENCE = PATIENCE if args.universal_patience is None else args.universal_patience
UNIVERSAL_LR_SCHEDULER = "cosine" if args.universal_cos_lr else "none"
UNIVERSAL_COSINE_T_MAX = args.universal_cos_t or UNIVERSAL_MAX_EPOCHS
TUNED_INITIAL_LR = float(
    args.tuned_lr if args.tuned_lr is not None else HYDRA_TRAIN_CFG.training.lr
)
TUNED_MAX_EPOCHS = args.tuned_max_epochs if args.tuned_max_epochs is not None else MAX_EPOCHS
TUNED_PATIENCE = args.tuned_patience if args.tuned_patience is not None else PATIENCE
TUNED_USE_EARLY_STOPPING = not args.tuned_no_early_stopping
TUNED_BATCH_SIZE = args.tuned_batch_size
TUNED_SHUFFLE = True
TUNED_FREEZE_FIRST_K = args.tuned_freeze_first_k
TUNED_RESET_FINAL_LAYER = args.tuned_reset_final_layer
TUNED_RESET_BN = args.tuned_reset_bn
TUNED_MONITOR = args.tuned_monitor
TUNED_LR_SCHEDULER = (
    "warmup_cosine" if args.warmup_cosine_lr
    else "onecycle" if args.onecycle_lr
    else "cosine" if args.tuned_cosine_lr
    else "plateau" if args.plateau_lr
    else "none"
)
TUNED_SOURCE_SELECTION = "target_val_eta" if args.tuned_source_val_eta else "val_loss"
TUNED_COSINE_T_MAX = args.cos_t
TUNED_COSINE_ETA_MIN = 1e-6
TUNED_WARMUP_EPOCHS = args.warmup_epochs
TUNED_WARMUP_START_FACTOR = args.warmup_start_factor
TUNED_ONECYCLE_PCT_START = args.onecycle_pct_start
TUNED_ONECYCLE_DIV_FACTOR = args.onecycle_div_factor
TUNED_ONECYCLE_FINAL_DIV_FACTOR = args.onecycle_final_div_factor
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


def label_value(value):
    return str(value).replace("-", "m").replace(".", "p")


def save_dir(root, seed, dropout=None, extra=None):
    name = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seed{seed}"
    if dropout is not None:
        name += f"_dropout{label_value(dropout)}"
    if extra:
        name += f"_{extra}"
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def git_dirty():
    status = git_value("status", "--porcelain")
    return "unavailable" if status == "unavailable" else bool(status)


def write_run_settings(run_dir, *, model_family, element, typ, seed, dropout, split, hidden_dims, model, source_info=None):
    cfg = model.cfg
    scheduler_min_lr = "none"
    if cfg.lr_scheduler in ("cosine", "warmup_cosine"):
        scheduler_min_lr = cfg.cosine_eta_min
    elif cfg.lr_scheduler == "plateau":
        scheduler_min_lr = cfg.plateau_min_lr

    sections = {
        "Run": {
            "model_family": model_family,
            "element": element,
            "spectrum_type": typ,
            "dataset": f"{element}_{typ}",
            "run_dir": run_dir,
            "script": Path(__file__).as_posix(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "Hardware / Seed": {
            "seed": seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            "torch_cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        },
        "Data": {
            "train_size": len(split.train.X),
            "val_size": len(split.val.X),
            "test_size": len(split.test.X),
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
        },
        "Architecture": {
            "hidden_dims": list(hidden_dims),
            "activation": "SiLU",
            "normalization": "BatchNorm1d",
            "dropout": dropout,
            "output_activation": "Softplus",
        },
        "Training": {
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "shuffle": cfg.shuffle,
            "monitor_metric": cfg.monitor_metric,
            "early_stopping_enabled": cfg.use_early_stopping,
            "early_stopping_patience": cfg.early_stopping_patience,
            "use_lr_finder": cfg.use_lr_finder,
        },
        "Optimizer": {
            "optimizer": getattr(model.model.optimizer, "__name__", str(model.model.optimizer)),
            "starting_learning_rate": cfg.initial_lr,
            "lr_finder_min_lr": cfg.min_lr,
        },
        "LR Scheduler": {
            "scheduler": cfg.lr_scheduler,
            "scheduler_min_lr": scheduler_min_lr,
            "cosine_t_max": cfg.cosine_t_max if cfg.lr_scheduler in ("cosine", "warmup_cosine") else "none",
            "plateau_factor": cfg.plateau_factor if cfg.lr_scheduler == "plateau" else "none",
            "plateau_patience": cfg.plateau_patience if cfg.lr_scheduler == "plateau" else "none",
            "plateau_monitor": cfg.monitor_metric if cfg.lr_scheduler == "plateau" else "none",
            "warmup_epochs": cfg.warmup_epochs if cfg.lr_scheduler == "warmup_cosine" else "none",
            "warmup_start_factor": cfg.warmup_start_factor if cfg.lr_scheduler == "warmup_cosine" else "none",
            "onecycle_max_lr": (cfg.onecycle_max_lr or cfg.initial_lr) if cfg.lr_scheduler == "onecycle" else "none",
            "onecycle_pct_start": cfg.onecycle_pct_start if cfg.lr_scheduler == "onecycle" else "none",
            "onecycle_div_factor": cfg.onecycle_div_factor if cfg.lr_scheduler == "onecycle" else "none",
            "onecycle_final_div_factor": cfg.onecycle_final_div_factor if cfg.lr_scheduler == "onecycle" else "none",
        },
        "Transfer Options": {
            "freeze_first_k_layers": TUNED_FREEZE_FIRST_K if model_family == "tunedUniversalXAS" else "none",
            "reset_final_layer": TUNED_RESET_FINAL_LAYER if model_family == "tunedUniversalXAS" else "none",
            "reset_batchnorm": TUNED_RESET_BN if model_family == "tunedUniversalXAS" else "none",
        },
        "Checkpointing": {
            "checkpoint_monitor": cfg.monitor_metric,
            "checkpoint_mode": "min",
            "save_top_k": 1,
            "save_last": True,
        },
        "Reproducibility": {
            "command": " ".join([sys.executable, *sys.argv]),
            "git_commit": git_value("rev-parse", "--short", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "hostname": socket.gethostname(),
            "working_directory": Path.cwd(),
        },
    }
    if source_info is not None:
        reordered = {}
        for title, values in sections.items():
            reordered[title] = values
            if title == "LR Scheduler":
                reordered["Fine-tuning Source"] = source_info
        sections = reordered

    lines = ["OmniXAS Run Settings", "====================", ""]
    for title, values in sections.items():
        lines.extend([title, "-" * len(title)])
        lines.extend(f"{key}: {value}" for key, value in values.items())
        lines.append("")
    (Path(run_dir) / "run_settings.txt").write_text("\n".join(lines), encoding="utf-8")


def reg(
    directory,
    dims,
    batch,
    *,
    initial_lr=INITIAL_LR,
    use_lr_finder=True,
    shuffle=False,
    max_epochs=MAX_EPOCHS,
    early_stopping_patience=PATIENCE,
    use_early_stopping=True,
    monitor_metric="val_loss",
    lr_scheduler="none",
    cosine_t_max=None,
    cosine_eta_min=1e-6,
    warmup_epochs=TUNED_WARMUP_EPOCHS,
    warmup_start_factor=TUNED_WARMUP_START_FACTOR,
    onecycle_pct_start=TUNED_ONECYCLE_PCT_START,
    onecycle_div_factor=TUNED_ONECYCLE_DIV_FACTOR,
    onecycle_final_div_factor=TUNED_ONECYCLE_FINAL_DIV_FACTOR,
):
    return XASBlockRegressor(
        directory=str(directory),
        overwrite_save_dir=False,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=list(dims),
        batch_size=batch,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        initial_lr=initial_lr,
        min_lr=MIN_LR,
        use_lr_finder=use_lr_finder,
        use_early_stopping=use_early_stopping,
        monitor_metric=monitor_metric,
        shuffle=shuffle,
        lr_scheduler=lr_scheduler,
        cosine_t_max=cosine_t_max,
        cosine_eta_min=cosine_eta_min,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        onecycle_pct_start=onecycle_pct_start,
        onecycle_div_factor=onecycle_div_factor,
        onecycle_final_div_factor=onecycle_final_div_factor,
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


def best_universal_source_by_val_loss(label, split):
    ckpts = sorted(run_root("universal").glob("paper_*/best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No UniversalXAS checkpoints found. Train UniversalXAS first.")
    best_ckpt = min(ckpts, key=checkpoint_val_loss)
    source_info = {
        "universal_source_selection": "val_loss",
        "universal_source_val_loss": checkpoint_val_loss(best_ckpt),
        "universal_source_target_val_eta": "not_computed",
    }
    print(
        f"Best UniversalXAS source for {label}: "
        f"universal_val_loss={source_info['universal_source_val_loss']:.8g} | {best_ckpt}",
        flush=True,
    )
    return best_ckpt.parent, source_info


def predict_with_checkpoint(ckpt, split):
    model = reg(ckpt.parent, UNIVERSAL_DIMS, 32, use_lr_finder=False, max_epochs=1)
    model.load("best")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = model.model.to(device).eval()
    X = np.asarray(split.val.X, dtype=np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 1024):
            xb = torch.tensor(X[start : start + 1024], dtype=torch.float32, device=device)
            preds.append(module(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32, copy=False)


def val_eta_for_checkpoint(ckpt, split):
    pred = np.asarray(predict_with_checkpoint(ckpt, split), dtype=np.float32)
    target = np.asarray(split.val.y, dtype=np.float32)
    train_y = np.asarray(split.train.y, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape} for {ckpt}")
    baseline = np.repeat(train_y.mean(axis=0, keepdims=True), len(target), axis=0)
    baseline_median = float(np.median(np.mean((target - baseline) ** 2, axis=1)))
    median_mse = float(np.median(np.mean((target - pred) ** 2, axis=1)))
    return baseline_median / median_mse


def best_universal_source_by_val_eta(label, split):
    ckpts = sorted(run_root("universal").glob("paper_*/best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No UniversalXAS checkpoints found. Train UniversalXAS first.")
    scored = [(val_eta_for_checkpoint(ckpt, split), checkpoint_val_loss(ckpt), ckpt) for ckpt in ckpts]
    best_eta, best_val_loss, best_ckpt = max(scored, key=lambda row: row[0])
    source_info = {
        "universal_source_selection": "target_val_eta",
        "universal_source_val_loss": best_val_loss,
        "universal_source_target_val_eta": best_eta,
    }
    print(
        f"Best UniversalXAS source for {label}: val_eta={best_eta:.6f}, "
        f"universal_val_loss={best_val_loss:.8g} | {best_ckpt}",
        flush=True,
    )
    return best_ckpt.parent, source_info


def default_tuned_cosine_t_max(typ):
    return TUNED_COSINE_T_MAX or (250 if typ == "FEFF" else 600)


def tuned_extra_label(batch_size, typ):
    parts = [f"lr{label_value(TUNED_INITIAL_LR)}"]
    if TUNED_LR_SCHEDULER == "cosine":
        parts.append(f"cosT{default_tuned_cosine_t_max(typ)}")
    elif TUNED_LR_SCHEDULER == "warmup_cosine":
        parts.append(f"warmcosT{default_tuned_cosine_t_max(typ)}")
        parts.append(f"warm{TUNED_WARMUP_EPOCHS}")
    elif TUNED_LR_SCHEDULER == "onecycle":
        parts.append("onecycle")
    elif TUNED_LR_SCHEDULER == "plateau":
        parts.append("plateau")
    return "_".join(parts)


def apply_tuned_transfer_options(pl_module):
    if not (TUNED_FREEZE_FIRST_K or TUNED_RESET_FINAL_LAYER or TUNED_RESET_BN):
        return
    linears = [m for m in pl_module.modules() if isinstance(m, torch.nn.Linear)]
    batchnorms = [m for m in pl_module.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    if TUNED_RESET_BN:
        for bn in batchnorms:
            bn.reset_running_stats()
        print(f"Reset {len(batchnorms)} BatchNorm running-stat objects", flush=True)
    if TUNED_RESET_FINAL_LAYER and linears:
        linears[-1].reset_parameters()
        print("Reset final Linear layer", flush=True)
    if TUNED_FREEZE_FIRST_K:
        for layer in linears[:TUNED_FREEZE_FIRST_K]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"Froze first {min(TUNED_FREEZE_FIRST_K, len(linears))} Linear layers", flush=True)


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
print(
    f"UniversalXAS: dropout={UNIVERSAL_DROPOUT}, lr={UNIVERSAL_INITIAL_LR}, "
    f"lr_finder={UNIVERSAL_USE_LR_FINDER}, max_epochs={UNIVERSAL_MAX_EPOCHS}, "
    f"patience={UNIVERSAL_PATIENCE}, monitor={args.universal_monitor}, "
    f"scheduler={UNIVERSAL_LR_SCHEDULER}, shuffle={args.universal_shuffle}",
    flush=True,
)
print(f"Tuned fine-tune LR from config/paper_hydra/train.yaml: {TUNED_INITIAL_LR}", flush=True)
print(f"Tuned dropouts: {TUNED_DROPOUTS}", flush=True)
print(f"Tuned max epochs: {TUNED_MAX_EPOCHS}", flush=True)
print(f"Tuned early stopping: {TUNED_USE_EARLY_STOPPING} | patience={TUNED_PATIENCE}", flush=True)
print(f"Tuned monitor metric: {TUNED_MONITOR}", flush=True)
print(f"Tuned LR scheduler: {TUNED_LR_SCHEDULER}", flush=True)
print(f"Tuned cosine T_max override: {TUNED_COSINE_T_MAX}", flush=True)
print(f"Tuned warmup epochs: {TUNED_WARMUP_EPOCHS}", flush=True)
print(f"Tuned warmup start factor: {TUNED_WARMUP_START_FACTOR}", flush=True)
print(f"Tuned OneCycle pct_start: {TUNED_ONECYCLE_PCT_START}", flush=True)
print(f"Tuned OneCycle div_factor: {TUNED_ONECYCLE_DIV_FACTOR}", flush=True)
print(f"Tuned OneCycle final_div_factor: {TUNED_ONECYCLE_FINAL_DIV_FACTOR}", flush=True)
print(f"Tuned UniversalXAS source selection: {TUNED_SOURCE_SELECTION}", flush=True)
print(f"Tuned batch override: {TUNED_BATCH_SIZE}", flush=True)
print(f"Tuned fine-tune DataLoader shuffle: {TUNED_SHUFFLE}", flush=True)
print(
    f"Tuned transfer options: freeze_first_k={TUNED_FREEZE_FIRST_K}, "
    f"reset_final_layer={TUNED_RESET_FINAL_LAYER}, reset_bn={TUNED_RESET_BN}",
    flush=True,
)

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
        XASBlock.DROPOUT = UNIVERSAL_DROPOUT
        d = save_dir(run_root("universal"), seed, UNIVERSAL_DROPOUT, f"lr{label_value(UNIVERSAL_INITIAL_LR)}_{UNIVERSAL_LR_SCHEDULER}")
        banner(job, 0, f"training UniversalXAS FEFF | seed={seed} | dir={d}")
        model = reg(
            d,
            UNIVERSAL_DIMS,
            32,
            initial_lr=UNIVERSAL_INITIAL_LR,
            use_lr_finder=UNIVERSAL_USE_LR_FINDER,
            shuffle=args.universal_shuffle,
            max_epochs=UNIVERSAL_MAX_EPOCHS,
            early_stopping_patience=UNIVERSAL_PATIENCE,
            monitor_metric=args.universal_monitor,
            lr_scheduler=UNIVERSAL_LR_SCHEDULER,
            cosine_t_max=UNIVERSAL_COSINE_T_MAX if UNIVERSAL_LR_SCHEDULER == "cosine" else None,
            cosine_eta_min=TUNED_COSINE_ETA_MIN,
        )
        write_run_settings(
            d,
            model_family="universalXAS",
            element="All",
            typ="FEFF",
            seed=seed,
            dropout=UNIVERSAL_DROPOUT,
            split=universal_split,
            hidden_dims=UNIVERSAL_DIMS,
            model=model,
        )
        model.fit(universal_split)
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
                model = reg(d, hparams["widths"], hparams["batch_size"])
                write_run_settings(
                    d,
                    model_family="expertXAS",
                    element=element,
                    typ=typ,
                    seed=seed,
                    dropout=DEFAULT_DROPOUT,
                    split=data,
                    hidden_dims=hparams["widths"],
                    model=model,
                )
                model.fit(data)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if "tuned" in models:
            label = f"{element} {typ} Tuned-UniversalXAS"
            source, source_info = (
                best_universal_source_by_val_eta(label, data)
                if TUNED_SOURCE_SELECTION == "target_val_eta"
                else best_universal_source_by_val_loss(label, data)
            )
            for seed in seeds:
                for dropout in TUNED_DROPOUTS:
                    job += 1
                    seed_everything(seed, workers=True)
                    XASBlock.DROPOUT = dropout
                    tuned_batch_size = TUNED_BATCH_SIZE or hparams["batch_size"]
                    extra = tuned_extra_label(tuned_batch_size, typ)
                    d = save_dir(run_root("tuned", element, typ), seed, dropout, extra)
                    tuned_cosine_t_max = default_tuned_cosine_t_max(typ) if TUNED_LR_SCHEDULER in ("cosine", "warmup_cosine") else None
                    banner(
                        job,
                        0,
                        f"fine-tuning {element} {typ} Tuned-UniversalXAS | seed={seed} | "
                        f"dropout={dropout} | lr={TUNED_INITIAL_LR} | bs={tuned_batch_size} | "
                        f"early_stop={TUNED_USE_EARLY_STOPPING} | patience={TUNED_PATIENCE} | "
                        f"monitor={TUNED_MONITOR} | scheduler={TUNED_LR_SCHEDULER} | "
                        f"cosine_t_max={tuned_cosine_t_max} | cosine_eta_min={TUNED_COSINE_ETA_MIN} | "
                        f"warmup_epochs={TUNED_WARMUP_EPOCHS} | onecycle_pct_start={TUNED_ONECYCLE_PCT_START} | "
                        f"shuffle={TUNED_SHUFFLE} | freeze_first_k={TUNED_FREEZE_FIRST_K} | "
                        f"reset_final={TUNED_RESET_FINAL_LAYER} | reset_bn={TUNED_RESET_BN} | dir={d}",
                    )
                    model = reg(
                        source,
                        UNIVERSAL_DIMS,
                        tuned_batch_size,
                        initial_lr=TUNED_INITIAL_LR,
                        use_lr_finder=False,
                        shuffle=TUNED_SHUFFLE,
                        max_epochs=TUNED_MAX_EPOCHS,
                        early_stopping_patience=TUNED_PATIENCE,
                        use_early_stopping=TUNED_USE_EARLY_STOPPING,
                        monitor_metric=TUNED_MONITOR,
                        lr_scheduler=TUNED_LR_SCHEDULER,
                        cosine_t_max=tuned_cosine_t_max,
                        cosine_eta_min=TUNED_COSINE_ETA_MIN,
                        warmup_epochs=TUNED_WARMUP_EPOCHS,
                        warmup_start_factor=TUNED_WARMUP_START_FACTOR,
                        onecycle_pct_start=TUNED_ONECYCLE_PCT_START,
                        onecycle_div_factor=TUNED_ONECYCLE_DIV_FACTOR,
                        onecycle_final_div_factor=TUNED_ONECYCLE_FINAL_DIV_FACTOR,
                    )
                    model.load("best")
                    apply_tuned_transfer_options(model.model)
                    model.cfg.directory = str(d)
                    write_run_settings(
                        d,
                        model_family="tunedUniversalXAS",
                        element=element,
                        typ=typ,
                        seed=seed,
                        dropout=dropout,
                        split=data,
                        hidden_dims=UNIVERSAL_DIMS,
                        model=model,
                        source_info=source_info,
                    )
                    model.fit(data)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

print("\nDone. Run: python tutorial_omnixas/evaluate_best_val_loss.py", flush=True)
