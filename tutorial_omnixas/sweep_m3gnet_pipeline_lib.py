"""M3GNet XAS large sweep library (4 encoders x 8 UniversalXAS x 16 tuned heads).

Stages:
    - Encoder stage: train scratch M3GNetXASEncoder variants, export their
      features. Runs sequentially in the notebook process.
    - Head stage: train SpectralHead variants on exported feature files.
      Runs in spawn-ed worker processes and works on a CPU-only box.

Import constraint: top-level imports must not touch dgl, matgl, pymatgen, or
lightning. ``omnixas.data.feff_graph`` and ``omnixas.model.m3gnet_xas`` are
imported lazily inside the encoder-stage functions because they load matgl.

Head jobs are spawn-safe: ``_head_job`` is a top-level function and job dicts
contain only picklable paths (as strings), strings, numbers, and plain dicts.

Fail-fast policy: missing files, unknown spec keys, incomplete head
directories, and non-canonical exports raise. There is no silent fallback or
checkpoint substitution.
"""
import os

# DGL must receive its backend before any import that could load dgl.
os.environ.setdefault("DGLBACKEND", "pytorch")

from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import time
import typing
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

SPECTRUM_DIM = 141
FEATURE_SCALE = 1000.0
FEFF_TASKS = ["Ti_FEFF", "V_FEFF", "Cr_FEFF", "Mn_FEFF", "Fe_FEFF", "Co_FEFF", "Ni_FEFF", "Cu_FEFF"]
SPLITS = ("train", "val", "test")

Job = typing.Dict[str, typing.Any]

# Keys accepted in head job specs; any other key raises.
_SPEC_KEYS = {
    "lr", "schedule", "cosine_t", "es_metric", "patience", "batch_size",
    "optimizer", "hidden_dims", "dropout", "head_dropout", "deriv_lambda",
    "warmup_epochs", "div_factor", "freeze_epochs", "lr_phase1", "lr_phase2",
    "weight_decay", "input_dim",
}

ENCODER_SPECS = {
    "enc_a_control": dict(blocks=3, feature_dim=64, cutoff=4.0, threebody_cutoff=4.0, dropout=0.10),
    "enc_b_deep4": dict(blocks=4, feature_dim=64, cutoff=4.0, threebody_cutoff=4.0, dropout=0.10),
    "enc_c_wide128": dict(blocks=3, feature_dim=128, cutoff=4.0, threebody_cutoff=4.0, dropout=0.10),
    "enc_d_cutoff5": dict(blocks=3, feature_dim=64, cutoff=5.0, threebody_cutoff=5.0, dropout=0.10),
}

UNIVERSAL_SPECS = {
    "u0_control": dict(optimizer="adam", hidden_dims=(500, 500, 550), dropout=0.10, deriv_lambda=0.0, lr=5e-4, schedule="plateau", es_metric="mean"),
    "u1_lr2e-4": dict(optimizer="adam", hidden_dims=(500, 500, 550), dropout=0.10, deriv_lambda=0.0, lr=2e-4, schedule="plateau", es_metric="mean"),
    "u2_lr1e-3": dict(optimizer="adam", hidden_dims=(500, 500, 550), dropout=0.10, deriv_lambda=0.0, lr=1e-3, schedule="plateau", es_metric="mean"),
    "u3_cosine": dict(optimizer="adam", hidden_dims=(500, 500, 550), dropout=0.10, deriv_lambda=0.0, lr=5e-4, schedule="cosine", cosine_t=800, es_metric="mean"),
    "u4_macro_eta": dict(optimizer="adam", hidden_dims=(500, 500, 550), dropout=0.10, deriv_lambda=0.0, lr=5e-4, schedule="plateau", es_metric="macro_eta"),
    "u5_256": dict(optimizer="adam", hidden_dims=(256, 256), dropout=0.10, deriv_lambda=0.0, lr=5e-4, schedule="plateau", es_metric="mean"),
    "u6_drop02": dict(optimizer="adam", hidden_dims=(500, 500, 550), dropout=0.20, deriv_lambda=0.0, lr=5e-4, schedule="plateau", es_metric="mean"),
    "u7_deriv": dict(optimizer="adam", hidden_dims=(500, 500, 550), dropout=0.10, deriv_lambda=0.02, lr=5e-4, schedule="plateau", es_metric="mean"),
}

TUNED_SPECS = {
    "t0_control": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-4, schedule="cosine", cosine_t=500),
    "t1_lr3e-5": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-5, schedule="cosine", cosine_t=500),
    "t2_plateau3e-4": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-4, schedule="plateau"),
    "t3_plateau3e-5": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-5, schedule="plateau"),
    "t4_plateau5e-4": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=5e-4, schedule="plateau"),
    "t5_plateau1e-3": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=1e-3, schedule="plateau"),
    "t6_warmup_cos": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-4, schedule="warmup_cos", warmup_epochs=10),
    "t7_onecycle": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-4, schedule="onecycle", div_factor=25.0),
    "t8_staged_unfreeze": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", schedule="staged", freeze_epochs=25, lr_phase1=3e-4, lr_phase2=3e-5),
    "t9_deriv": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", deriv_lambda=0.02, lr=3e-4, schedule="plateau"),
    "t10_adamw": dict(optimizer="adamw", hidden_dims=(500, 500, 550), es_metric="median", weight_decay=1e-4, lr=3e-4, schedule="plateau"),
    "t11_drop00": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", head_dropout=0.0, lr=3e-4, schedule="plateau"),
    "t12_drop02": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", head_dropout=0.2, lr=3e-4, schedule="plateau"),
    "t13_cosT1000": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-4, schedule="cosine", cosine_t=1000),
    "t14_es30": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-4, schedule="plateau", patience=30),
    "t15_bs512": dict(optimizer="adam", hidden_dims=(500, 500, 550), es_metric="median", lr=3e-4, schedule="plateau", batch_size=512),
}


class SpectralHead(nn.Sequential):
    """Nonnegative XAS head: per hidden width Linear -> BatchNorm1d -> SiLU ->
    Dropout, final Linear -> Softplus. Structure and nn.Sequential integer
    layer naming match XASSpectralHead so state_dict keys align."""

    def __init__(self, input_dim: int = 64, hidden_dims: tuple = (500, 500, 550), output_dim: int = 141, dropout: float = 0.10) -> None:
        layers: list[nn.Module] = []
        input_width = int(input_dim)
        for hidden_width in hidden_dims:
            layers.extend([
                nn.Linear(input_width, int(hidden_width)),
                nn.BatchNorm1d(int(hidden_width)),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            input_width = int(hidden_width)
        layers.extend([nn.Linear(input_width, int(output_dim)), nn.Softplus()])
        super().__init__(*layers)


def _load_splits(features_dir: Path, task: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load {task}_{split}_{X,y}.txt for the 3 splits of one task and validate shapes."""
    if task not in FEFF_TASKS:
        raise ValueError(f"Unknown FEFF task: {task}")
    features_dir = Path(features_dir)
    missing = []
    for split in SPLITS:
        for suffix in ("X", "y"):
            path = features_dir / f"{task}_{split}_{suffix}.txt"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing feature files for " + task + ":\n" + "\n".join(missing))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        X = np.atleast_2d(np.loadtxt(features_dir / f"{task}_{split}_X.txt", dtype=np.float32))
        y = np.atleast_2d(np.loadtxt(features_dir / f"{task}_{split}_y.txt", dtype=np.float32))
        if X.ndim != 2:
            raise ValueError(f"Feature matrix is not 2D for {task} {split}: {X.shape}")
        if y.shape[1] != SPECTRUM_DIM:
            raise ValueError(f"Target matrix must have {SPECTRUM_DIM} columns for {task} {split}: {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Row count mismatch for {task} {split}: X={X.shape[0]}, y={y.shape[0]}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite feature or target value for {task} {split}")
        out[split] = (X, y)
    return out


def canonical_y(repo_root: Path, task: str, split: str) -> np.ndarray:
    path = Path(repo_root) / "tutorial_omnixas" / "ml_data" / f"{task}_{split}_y.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical target file: {path}")
    return np.atleast_2d(np.loadtxt(path, dtype=np.float32))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    except Exception:
        return ""
    return out.stdout.strip()


def baselines(repo_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-task median per-spectrum MSE of train/val targets vs the train mean
    spectrum of the same task (pipeline logic)."""
    data = Path(repo_root) / "tutorial_omnixas" / "ml_data"
    train, val = [], []
    for task in FEFF_TASKS:
        ty = np.atleast_2d(np.loadtxt(data / f"{task}_train_y.txt", dtype=np.float32))
        mean = ty.mean(0)
        train.append(float(np.median(((ty - mean) ** 2).mean(1))))
        vy = np.atleast_2d(np.loadtxt(data / f"{task}_val_y.txt", dtype=np.float32))
        val.append(float(np.median(((vy - mean) ** 2).mean(1))))
    return torch.tensor(train), torch.tensor(val)


def balanced_universal_arrays(splits_by_task: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample each task train split with replacement up to the max task count,
    concatenate; val = concatenation of all val splits (pipeline logic)."""
    n = max(len(splits_by_task[task]["train"][0]) for task in FEFF_TASKS)
    rng = np.random.default_rng(seed)
    train_X, train_y = [], []
    for task in FEFF_TASKS:
        X, y = splits_by_task[task]["train"]
        index = rng.integers(len(X), size=n)
        train_X.append(X[index])
        train_y.append(y[index])
    val_X = np.concatenate([splits_by_task[task]["val"][0] for task in FEFF_TASKS], axis=0)
    val_y = np.concatenate([splits_by_task[task]["val"][1] for task in FEFF_TASKS], axis=0)
    return np.concatenate(train_X, axis=0), np.concatenate(train_y, axis=0), val_X, val_y


def _batched_mse(head: nn.Module, X: np.ndarray, y: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    """Per-spectrum MSE of head on X (inference mode, internal batching)."""
    head.eval()
    tX = torch.as_tensor(X, dtype=torch.float32, device=device)
    preds = []
    with torch.inference_mode():
        for i in range(0, tX.shape[0], batch):
            preds.append(head(tX[i : i + batch]).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return np.mean((pred - y) ** 2, axis=1)


def task_val_metrics(head: nn.Module, X: np.ndarray, y: np.ndarray, y_train: np.ndarray, device, batch: int = 4096) -> dict:
    """Val metrics of one task; eta baseline is the train-mean spectrum of the
    same task (pipeline evaluate_head formula)."""
    mse = _batched_mse(head, X, y, device, batch)
    baseline = float(np.median(np.mean((y - y_train.mean(0)) ** 2, axis=1)))
    median = float(np.median(mse))
    return {"val_mse": float(mse.mean()), "val_median_mse": median, "val_eta": baseline / max(median, 1e-12)}


def per_element_val_metrics(head: nn.Module, splits_by_task: dict, device, batch: int = 4096) -> dict[str, dict]:
    """Per-task val metrics. splits_by_task maps each FEFF task to
    (X_eval, y_eval, y_train)."""
    out = {}
    for task in FEFF_TASKS:
        if task not in splits_by_task:
            raise ValueError(f"Missing task for per-element metrics: {task}")
        X, y, y_train = splits_by_task[task]
        out[task] = task_val_metrics(head, X, y, y_train, device, batch)
    return out


def macro_val_eta(per_element: dict) -> float:
    """Unweighted mean of the 8 per-element val etas."""
    missing = [task for task in FEFF_TASKS if task not in per_element]
    if missing:
        raise ValueError("Missing tasks for macro val eta: " + str(missing))
    return float(np.mean([per_element[task]["val_eta"] for task in FEFF_TASKS]))


def _build_optimizer(spec: dict, params, device) -> torch.optim.Optimizer:
    name = spec.get("optimizer", "adam")
    if name not in ("adam", "adamw"):
        raise ValueError(f"Unknown optimizer: {name}")
    cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
    return cls(params, lr=float(spec["lr"]), weight_decay=float(spec.get("weight_decay", 0.0)), fused=device.type == "cuda")


def _make_scheduler(spec: dict, opt: torch.optim.Optimizer, total_epochs: int, epoch_offset: int = 0):
    """Build the head LR scheduler. plateau steps on val mean mse (caller);
    the others step once per epoch; staged is handled in the epoch loop."""
    schedule = spec.get("schedule", "plateau")
    if schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=16, min_lr=1e-6)
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(spec["cosine_t"]), eta_min=1e-6)
    if schedule == "warmup_cos":
        warmup = int(spec["warmup_epochs"])
        base_lr = float(spec["lr"])
        eta_min = 1e-6
        span = max(1, total_epochs - warmup)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return 0.1 + 0.9 * epoch / warmup
            t = epoch - warmup
            return eta_min / base_lr + 0.5 * (1.0 + math.cos(math.pi * t / span)) * (1.0 - eta_min / base_lr)

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if schedule == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(spec["lr"]),
            div_factor=float(spec.get("div_factor", 25.0)),
            total_steps=total_epochs,
            steps_per_epoch=1,
            anneal_strategy="cos",
        )
    if schedule == "staged":
        return None
    raise ValueError(f"Unknown schedule: {schedule}")


def _staged_params(head: nn.Sequential) -> list:
    """Params of the last hidden block (Linear, BN, SiLU, Dropout) plus the
    output Linear, found by walking head.children()."""
    children = list(head.children())
    if len(children) < 6 or not isinstance(children[-1], nn.Softplus) or not isinstance(children[-2], nn.Linear):
        raise ValueError("SpectralHead tail must be (Linear, Softplus); cannot apply staged schedule")
    tail = children[-6:-1]
    params = [p for module in tail for p in module.parameters()]
    if not params:
        raise ValueError("Staged schedule found no trainable parameters")
    return params


def _validate_spec(stage: str, variant: str, spec: dict) -> None:
    unknown = sorted(set(spec) - _SPEC_KEYS)
    if unknown:
        raise ValueError(f"Unknown spec keys for {stage}/{variant}: {unknown}")
    schedule = spec.get("schedule", "plateau")
    if schedule not in ("plateau", "cosine", "warmup_cos", "onecycle", "staged"):
        raise ValueError(f"Unknown schedule: {schedule}")
    es_metric = spec.get("es_metric")
    if es_metric not in ("mean", "median", "macro_eta"):
        raise ValueError(f"Unknown es_metric: {es_metric}")
    if "hidden_dims" not in spec:
        raise ValueError(f"Spec missing hidden_dims for {stage}/{variant}")
    if schedule == "staged":
        for key in ("freeze_epochs", "lr_phase1", "lr_phase2"):
            if key not in spec:
                raise ValueError(f"Staged schedule spec missing {key}")
    elif spec.get("lr") is None:
        raise ValueError(f"Spec missing lr for {stage}/{variant}")
    if schedule == "cosine" and "cosine_t" not in spec:
        raise ValueError("Cosine schedule spec missing cosine_t")
    if schedule == "warmup_cos" and "warmup_epochs" not in spec:
        raise ValueError("warmup_cos schedule spec missing warmup_epochs")
    if spec.get("optimizer", "adam") not in ("adam", "adamw"):
        raise ValueError(f"Unknown optimizer: {spec.get('optimizer')}")


def _head_job(job: Job) -> dict:
    """Spawn-safe head training worker. See module docstring for the job dict
    contract. Writes best.pt plus a metrics.json sidecar and returns the
    sidecar dict. job["epochs"] is authoritative; the spec may override the
    job patience/batch_size (variant-specific: t14_es30, t15_bs512)."""
    started = time.time()
    for key in ("run_dir", "features_dir", "repo_root", "stage", "enc", "variant", "seed", "epochs", "patience", "batch_size", "spec", "log_path"):
        if key not in job:
            raise ValueError(f"Head job missing key: {key}")
    run_dir = Path(job["run_dir"])
    features_dir = Path(job["features_dir"])
    repo_root = Path(job["repo_root"])
    stage = job["stage"]
    enc = job["enc"]
    variant = job["variant"]
    task = job.get("task")
    seed = int(job["seed"])
    spec = dict(job["spec"])
    log_path = Path(job["log_path"])

    if stage not in ("universal", "tuned"):
        raise ValueError(f"Unknown stage: {stage}")
    if stage == "universal" and task is not None:
        raise ValueError("Universal head job must not set task")
    if stage == "tuned" and not task:
        raise ValueError("Tuned head job requires task")
    if task is not None and task not in FEFF_TASKS:
        raise ValueError(f"Unknown FEFF task: {task}")
    _validate_spec(stage, variant, spec)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(job.get("torch_threads", 1))))

    out_dir = run_dir / "heads" / stage / variant
    if stage == "tuned":
        out_dir = out_dir / task
    checkpoint = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.json"
    if checkpoint.is_file():
        if metrics_path.is_file():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"Incomplete head directory (best.pt without metrics.json): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = spec.get("dropout", 0.10) if stage == "universal" else spec.get("head_dropout", 0.10)
    head = SpectralHead(
        input_dim=int(spec.get("input_dim", 64)),
        hidden_dims=tuple(int(h) for h in spec["hidden_dims"]),
        output_dim=SPECTRUM_DIM,
        dropout=dropout,
    )
    source_state_path = job.get("source_state_path")
    source_sha256 = None
    if source_state_path:
        source_path = Path(source_state_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source checkpoint: {source_path}")
        state = torch.load(source_path, map_location="cpu", weights_only=False)
        if "state_dict" not in state:
            raise ValueError(f"Source checkpoint is missing state_dict: {source_path}")
        head.load_state_dict(state["state_dict"], strict=True)
        source_sha256 = sha256_file(source_path)
    head.to(device)

    if stage == "universal":
        splits = {t: _load_splits(features_dir, t) for t in FEFF_TASKS}
        train_X, train_y, val_X, val_y = balanced_universal_arrays(splits, seed)
        eval_pairs = {t: (splits[t]["val"][0], splits[t]["val"][1], splits[t]["train"][1]) for t in FEFF_TASKS}
    else:
        splits = {task: _load_splits(features_dir, task)}
        train_X, train_y = splits[task]["train"]
        val_X, val_y = splits[task]["val"]
        eval_pairs = {task: (val_X, val_y, train_y)}

    epochs = int(job["epochs"])
    patience = int(spec.get("patience", job["patience"]))
    batch_size = int(spec.get("batch_size", job["batch_size"]))
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError(f"Invalid head job settings: epochs={epochs}, patience={patience}, batch_size={batch_size}")
    schedule = spec.get("schedule", "plateau")
    es_metric = spec["es_metric"]
    deriv_lambda = float(spec.get("deriv_lambda", 0.0))

    train_X_t = torch.as_tensor(train_X, device=device)
    train_y_t = torch.as_tensor(train_y, device=device)

    if schedule == "staged":
        opt = torch.optim.Adam(_staged_params(head), lr=float(spec["lr_phase1"]), fused=device.type == "cuda")
        sched = None
    else:
        opt = _build_optimizer(spec, head.parameters(), device)
        sched = _make_scheduler(spec, opt, epochs)

    maximize = es_metric == "macro_eta"
    best = float("-inf") if maximize else float("inf")
    stale = 0
    best_epoch = -1
    stopped_epoch = None
    label = "/".join((stage, enc, variant) + ((task,) if task else ()))

    def evaluate(module: nn.Module) -> tuple[float, float]:
        """Return (val mean mse, selection metric) for the given module."""
        module.eval()
        with torch.inference_mode():
            mse = _batched_mse(module, val_X, val_y, device, batch=4096)
        val_mean = float(mse.mean())
        if es_metric == "mean":
            return val_mean, val_mean
        if es_metric == "median":
            return val_mean, float(np.median(mse))
        return val_mean, macro_val_eta(per_element_val_metrics(module, eval_pairs, device, batch=4096))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    progress = tqdm(range(epochs), desc=label, unit="epoch", file=log_file)
    try:
        for epoch in progress:
            if schedule == "staged" and epoch == int(spec["freeze_epochs"]):
                opt = torch.optim.Adam(head.parameters(), lr=float(spec["lr_phase2"]), fused=device.type == "cuda")
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - int(spec["freeze_epochs"])), eta_min=1e-6)
                tqdm.write(f"[{label}] staged phase 2: all parameters at lr={spec['lr_phase2']}", file=log_file)
            head.train()
            indices = torch.randperm(len(train_X_t), device=device)
            for index in indices.split(batch_size):
                opt.zero_grad(set_to_none=True)
                pred = head(train_X_t[index])
                loss = (pred - train_y_t[index]).square().mean()
                if deriv_lambda > 0:
                    loss = loss + deriv_lambda * (torch.diff(pred, dim=1) - torch.diff(train_y_t[index], dim=1)).square().mean()
                loss.backward()
                opt.step()
            val, sel = evaluate(head)
            improved = sel > best if maximize else sel < best
            if improved:
                best, stale, best_epoch = sel, 0, epoch
                torch.save({"state_dict": head.state_dict(), "epoch": epoch, "val_loss": val}, checkpoint)
            else:
                stale += 1
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val)
            elif sched is not None:
                sched.step()
            progress.set_postfix(val=f"{val:.2e}", best=f"{best:.2e}", lr=f"{opt.param_groups[0]['lr']:.2e}")
            if stale >= patience:
                stopped_epoch = epoch + 1
                break
    finally:
        progress.close()
    if stopped_epoch is not None:
        tqdm.write(f"[{label}] early stopping at epoch {stopped_epoch}/{epochs}, best at epoch {best_epoch}", file=log_file)
    if not checkpoint.is_file():
        log_file.close()
        raise RuntimeError(f"Head training produced no checkpoint: {out_dir}")
    epochs_run = stopped_epoch if stopped_epoch is not None else epochs

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in saved:
        raise ValueError(f"Checkpoint is missing state_dict: {checkpoint}")
    head.load_state_dict(saved["state_dict"], strict=True)
    _, final_selection = evaluate(head)
    if stage == "universal":
        val_metrics = per_element_val_metrics(head, eval_pairs, device, batch=4096)
        macro = macro_val_eta(val_metrics)
    else:
        val_metrics = {task: task_val_metrics(head, val_X, val_y, train_y, device, batch=4096)}
        macro = None

    feature_sha256 = {}
    for t in (FEFF_TASKS if stage == "universal" else [task]):
        for split in ("train", "val"):
            for suffix in ("X", "y"):
                p = features_dir / f"{t}_{split}_{suffix}.txt"
                if not p.is_file():
                    raise FileNotFoundError(f"Missing feature file for hashing: {p}")
                feature_sha256[str(p)] = sha256_file(p)

    spec = dict(spec)
    spec["hidden_dims"] = list(spec["hidden_dims"])
    metrics = {
        "stage": stage,
        "enc": enc,
        "variant": variant,
        "task": task,
        "seed": seed,
        "spec": spec,
        "source_state_path": str(source_state_path) if source_state_path else None,
        "source_sha256": source_sha256,
        "best_epoch": best_epoch,
        "best_metric_name": {"mean": "val_mean_mse", "median": "val_median_mse", "macro_eta": "macro_val_eta"}[es_metric],
        "best_metric_value": float(best),
        "final_selection_metric": float(final_selection),
        "val_metrics": val_metrics,
        "macro_val_eta": macro,
        "feature_sha256": feature_sha256,
        "git_sha": git_sha(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
        "wall_seconds": round(time.time() - started, 3),
        "epochs_run": epochs_run,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tqdm.write(f"[{label}] best epoch {best_epoch}, {metrics['best_metric_name']}={best:.4g}, final_selection_metric={final_selection:.4g}", file=log_file)
    for t in sorted(val_metrics):
        tqdm.write(f"[{label}] {t} val_eta={val_metrics[t]['val_eta']:.3f} val_median_mse={val_metrics[t]['val_median_mse']:.4e}", file=log_file)
    if macro is not None:
        tqdm.write(f"[{label}] macro_val_eta={macro:.3f}", file=log_file)
    tqdm.write(f"[{label}] done, checkpoint: {checkpoint}", file=log_file)
    log_file.close()
    return metrics


def run_jobs_parallel(jobs: list[Job], max_workers: int) -> list[dict]:
    """Run _head_job over jobs with spawn-ed processes (CUDA-safe), in order.
    If any job raises, pending jobs are cancelled and the first exception is
    re-raised."""
    if not jobs:
        raise ValueError("run_jobs_parallel requires at least one job")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = [executor.submit(_head_job, job) for job in jobs]
        try:
            return [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise


def ensure_file_limit(workers: int) -> None:
    """Raise if the process cannot open enough file descriptors for DataLoader
    workers (Linux only)."""
    if workers <= 0 or os.name != "posix":
        return
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        soft = hard
    need = 256 + 32 * workers
    proc_fd = Path("/proc/self/fd")
    open_fds = len(list(proc_fd.iterdir())) if proc_fd.is_dir() else 0
    if open_fds + need > soft:
        raise RuntimeError(
            f"Not enough open file descriptors for {workers} DataLoader workers: "
            f"{open_fds} open, limit {soft}, need about {need}. "
            "Restart the notebook kernel, raise the shell limit (ulimit -n 4096), or lower num_workers."
        )


def _check_canonical_data(repo_root: Path) -> dict[str, int]:
    """Pipeline preflight data checks: file existence, row/shape alignment,
    duplicate IDs, finiteness, material leakage. Returns train row counts."""
    root = Path(repo_root)
    data = root / "tutorial_omnixas" / "ml_data"
    ids_dir = root / "tutorial_omnixas" / "material_id_and_site"
    missing = []
    for task in FEFF_TASKS:
        for split in SPLITS:
            for path in (data / f"{task}_{split}_X.txt", data / f"{task}_{split}_y.txt", ids_dir / f"{task}_{split}.txt"):
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing FEFF data files:\n" + "\n".join(missing[:12]))
    counts: dict[str, int] = {}
    material_sets: dict[tuple[str, str], set] = {}
    for task in FEFF_TASKS:
        for split in SPLITS:
            X = np.atleast_2d(np.loadtxt(data / f"{task}_{split}_X.txt", dtype=np.float32))
            y = canonical_y(root, task, split)
            rows = [line.strip() for line in (ids_dir / f"{task}_{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != len(set(rows)):
                raise ValueError(f"Duplicate full IDs in {ids_dir / f'{task}_{split}.txt'}")
            if X.ndim != 2 or X.shape[0] != len(rows) or y.shape != (len(rows), SPECTRUM_DIM):
                raise ValueError(f"Row/shape mismatch for {task} {split}: X={X.shape}, y={y.shape}, IDs={len(rows)}")
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite feature or target value for {task} {split}")
            material_sets[(task, split)] = {row.rsplit("_", 1)[0] for row in rows}
            if split == "train":
                counts[task] = len(rows)
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = material_sets[(task, left)] & material_sets[(task, right)]
            if overlap:
                raise ValueError(f"Material leakage in {task}: {left}/{right}, example {next(iter(overlap))}")
    return counts


def encoder_preflight(repo_root: Path, raw_root: Path) -> None:
    """Encoder-stage preflight. Imports matgl/dgl lazily; the head stage never
    calls this."""
    from omnixas.data import feff_graph as feff

    feff.patch_matgl_gpu_constants()
    raw = Path(raw_root)
    if not raw.is_dir():
        raise FileNotFoundError(f"Missing raw FEFF structure root: {raw}. Set OMNIXAS_DATA_ROOT.")
    _check_canonical_data(Path(repo_root))
    feff.validate_raw_structures(Path(repo_root), raw, list(feff.FEFF_TASKS))


def train_encoder(
    name: str,
    spec: dict,
    run_dir: Path,
    repo_root: Path,
    raw_root: Path,
    seed: int,
    *,
    epochs: int,
    rows_per_element: int,
    eval_batch: int,
    num_workers: int,
    lr: float = 1e-3,
) -> Path:
    """Train one scratch encoder (sequential, notebook process). Reuses an
    existing best_encoder.ckpt instead of retraining. Mid-fit resume via
    encoder_checkpoints/last.ckpt; complete runs reuse best_encoder.ckpt."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from omnixas.data import feff_graph as feff
    from omnixas.model.m3gnet_xas import M3GNetXASEncoder

    class LitScratchEncoder(pl.LightningModule):
        """Scratch M3GNet encoder Lightning module mirroring the pipeline
        LitScratch: the encoder is trained jointly with a full 500/500/550
        SpectralHead at the 1000x feature scale. Only encoder weights are
        kept in best_encoder.ckpt."""

        def __init__(self, encoder: nn.Module, train_base: torch.Tensor, val_base: torch.Tensor, lr: float, feature_dim: int) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = SpectralHead(
                input_dim=feature_dim,
                hidden_dims=(500, 500, 550),
                output_dim=SPECTRUM_DIM,
                dropout=0.10,
            )
            self.lr = lr
            self.register_buffer("train_base", train_base)
            self.register_buffer("val_base", val_base)
            self.val_mse = []
            self.val_task = []

        def step(self, batch, stage: str) -> torch.Tensor:
            graph = batch["graph"].to(self.device)
            line_graph = batch["line_graph"].to(self.device)
            site = batch["site"].to(self.device)
            pred = self.head(self.encoder(graph, line_graph, site) * FEATURE_SCALE)
            y = batch["y"].to(self.device)
            task = batch["task"].to(self.device)
            mse = ((pred - y) ** 2).mean(1)
            base = self.train_base if stage == "train" else self.val_base
            loss = (mse / base[task].clamp_min(1e-12)).mean() + 0.02 * (
                torch.diff(pred, dim=1) - torch.diff(y, dim=1)
            ).square().mean()
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
            if stage == "val":
                self.val_mse.append(mse.detach())
                self.val_task.append(task.detach())
            return loss

        def training_step(self, batch, _) -> torch.Tensor:
            return self.step(batch, "train")

        def on_validation_epoch_start(self) -> None:
            self.val_mse = []
            self.val_task = []

        def validation_step(self, batch, _) -> torch.Tensor:
            return self.step(batch, "val")

        def on_validation_epoch_end(self) -> None:
            if self.trainer.sanity_checking or not self.val_mse:
                return
            mses, tasks = torch.cat(self.val_mse), torch.cat(self.val_task)
            rel = [
                mses[tasks == i].median() / self.val_base[i].clamp_min(1e-12)
                for i in range(len(FEFF_TASKS))
                if (tasks == i).any()
            ]
            if len(rel) != len(FEFF_TASKS):
                raise RuntimeError("Validation lacks one or more FEFF elements")
            self.log("val_balanced_rel_mse", torch.stack(rel).mean(), on_epoch=True, prog_bar=True)

        def configure_optimizers(self):
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=1e-5,
                fused=self.device.type == "cuda",
            )
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=16, min_lr=1e-6)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_balanced_rel_mse"},
            }

    run_dir = Path(run_dir)
    root = Path(repo_root)
    raw = Path(raw_root)
    if not raw.is_dir():
        raise FileNotFoundError(f"Missing raw FEFF structure root: {raw}. Set OMNIXAS_DATA_ROOT.")
    _check_canonical_data(root)
    feff.patch_matgl_gpu_constants()
    ensure_file_limit(num_workers)
    checkpoint_path = run_dir / "best_encoder.ckpt"
    if checkpoint_path.is_file():
        print(f"Encoder {name}: reusing existing checkpoint: {checkpoint_path}", flush=True)
        return checkpoint_path
    run_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed, workers=True)
    encoder = M3GNetXASEncoder(
        dropout=spec["dropout"],
        feature_dim=spec["feature_dim"],
        blocks=spec["blocks"],
        cutoff=spec["cutoff"],
        threebody_cutoff=spec["threebody_cutoff"],
    )
    train_base, val_base = baselines(root)
    collate = feff.CollateGraphs(encoder)
    train_ds = feff.FEFFDataset(root, raw, list(feff.FEFF_TASKS), "train")
    task_counts = {task: sum(row[0] == task for row in train_ds.rows) for task in FEFF_TASKS}
    rows = min(rows_per_element, min(task_counts.values()))
    sampler = feff.BalancedTaskBatchSampler(train_ds.rows, rows, seed)
    graph_loader_kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        graph_loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, **graph_loader_kwargs)
    val_loader = DataLoader(feff.FEFFDataset(root, raw, list(feff.FEFF_TASKS), "val"), batch_size=eval_batch, collate_fn=collate, **graph_loader_kwargs)
    print(f"Encoder {name} batch sizes: train={sampler.batch_size} ({rows} rows per element), eval={eval_batch}", flush=True)
    model = LitScratchEncoder(encoder, train_base, val_base, lr, feature_dim=spec["feature_dim"])
    cb = ModelCheckpoint(
        run_dir / "encoder_checkpoints",
        filename="best-{epoch:03d}-{val_balanced_rel_mse:.5f}",
        monitor="val_balanced_rel_mse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    precision = "bf16-mixed" if torch.cuda.is_available() else "32-true"
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=[cb, EarlyStopping(monitor="val_balanced_rel_mse", patience=60, mode="min")],
        logger=CSVLogger(str(run_dir), name="encoder_logs"),
        log_every_n_steps=10,
    )
    last_ckpt = run_dir / "encoder_checkpoints" / "last.ckpt"
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(last_ckpt) if last_ckpt.is_file() else None)
    if not cb.best_model_path:
        raise RuntimeError("Encoder training produced no validation checkpoint")
    state = torch.load(cb.best_model_path, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise ValueError(f"Encoder checkpoint is missing state_dict: {cb.best_model_path}")
    enc_state = {k.removeprefix("encoder."): v for k, v in state["state_dict"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state, strict=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "name": name,
            "spec": spec,
            "seed": seed,
            "epoch": state.get("epoch"),
            "val_balanced_rel_mse": state.get("val_balanced_rel_mse"),
        },
        checkpoint_path,
    )
    print(f"Encoder {name}: saved {checkpoint_path}", flush=True)
    return checkpoint_path


def export_features(ckpt_path: Path, run_dir: Path, repo_root: Path, raw_root: Path, spec: dict, eval_batch: int) -> None:
    """Export {task}_{split}_{X,y}.txt features for missing splits, then
    validate every split (shape, finiteness, y exactly equal to canonical)."""
    from torch.utils.data import DataLoader

    from omnixas.data import feff_graph as feff
    from omnixas.model.m3gnet_xas import M3GNetXASEncoder

    ckpt_path = Path(ckpt_path)
    run_dir = Path(run_dir)
    root = Path(repo_root)
    raw = Path(raw_root)
    if not raw.is_dir():
        raise FileNotFoundError(f"Missing raw FEFF structure root: {raw}. Set OMNIXAS_DATA_ROOT.")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise ValueError(f"Encoder checkpoint is missing state_dict: {ckpt_path}")
    encoder = M3GNetXASEncoder(
        dropout=spec["dropout"],
        feature_dim=spec["feature_dim"],
        blocks=spec["blocks"],
        cutoff=spec["cutoff"],
        threebody_cutoff=spec["threebody_cutoff"],
    )
    encoder.load_state_dict(state["state_dict"], strict=True)
    encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    features = run_dir / "features"
    features.mkdir(parents=True, exist_ok=True)
    expected = [(task, split) for task in FEFF_TASKS for split in SPLITS]
    missing = [
        key
        for key in expected
        if not (features / f"{key[0]}_{key[1]}_X.txt").is_file() or not (features / f"{key[0]}_{key[1]}_y.txt").is_file()
    ]
    if not missing:
        print("Feature export: all splits present; validating existing files only.")
    else:
        collate = feff.CollateGraphs(encoder)
        with torch.inference_mode():
            for task, split in tqdm(missing, desc="M3GNet feature export", unit="split"):
                loader = DataLoader(feff.FEFFDataset(root, raw, [task], split), batch_size=eval_batch, collate_fn=collate)
                xs, ys = [], []
                for batch in loader:
                    xs.append(encoder(batch["graph"].to(device), batch["line_graph"].to(device), batch["site"].to(device)).mul(FEATURE_SCALE).cpu().numpy())
                    ys.append(batch["y"].numpy())
                X = np.concatenate(xs, axis=0)
                y = np.concatenate(ys, axis=0)
                np.savetxt(features / f"{task}_{split}_X.txt", X)
                np.savetxt(features / f"{task}_{split}_y.txt", y)
    for task, split in expected:
        X = np.atleast_2d(np.loadtxt(features / f"{task}_{split}_X.txt", dtype=np.float32))
        y = np.atleast_2d(np.loadtxt(features / f"{task}_{split}_y.txt", dtype=np.float32))
        if X.shape != (y.shape[0], spec["feature_dim"]) or y.shape[1] != SPECTRUM_DIM:
            raise ValueError(f"Invalid exported feature shape for {task} {split}: X={X.shape}, y={y.shape}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite exported feature or target for {task} {split}")
        if not np.array_equal(y, canonical_y(root, task, split)):
            raise ValueError(f"Exported targets do not exactly match canonical targets: {task} {split}")
        print(f"  {task} {split}: {X.shape[0]} rows, feature_dim={X.shape[1]}")
    print(f"Feature export complete: {len(expected)} splits under {features}")
