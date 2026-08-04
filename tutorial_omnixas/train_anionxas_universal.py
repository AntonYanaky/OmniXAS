#!/usr/bin/env python3
"""Train and evaluate AnionXAS UniversalXAS heads.

Example: ``python tutorial_omnixas/train_anionxas_universal.py --gpu 0``
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

# Set the device before importing torch or Lightning.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from omnixas.model.training import PlModule
from omnixas.model.xasblock import XASBlock

INPUT_DIM = 64
OUTPUT_DIM = 200
HIDDEN_DIMS = [500, 500, 550]
SEED = 42
DEFAULT_RUN_NAME = "anionxas_universal_first_benchmark"
DEFAULT_TUNED_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
TUNED_SETTINGS = [
    {
        "name": "e2e_head_hparams",
        "lr": 7e-4,
        "dropout": 0.25,
        "max_epochs": 300,
        "patience": 30,
        "scheduler": "cosine",
        "cosine_t": 300,
        "batch_size": 32,
    },
    {
        "name": "cosT500_lr3e-4_do0p1",
        "lr": 3e-4,
        "dropout": 0.1,
        "max_epochs": 1000,
        "patience": 25,
        "scheduler": "cosine",
        "cosine_t": 500,
        "batch_size": 32,
    },
]
UNIVERSAL_HPARAMS = {**TUNED_SETTINGS[0], "monitor": "val_median_mse"}

OMNIXAS_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
PAPER_UNIVERSAL_TEST_ETA = {
    "Ti": 4.19,
    "V": 5.19,
    "Cr": 7.13,
    "Mn": 13.15,
    "Fe": 6.04,
    "Co": 9.58,
    "Ni": 6.43,
    "Cu": 2.75,
}
PAPER_TUNED_TEST_ETA = {
    "Ti": 7.63,
    "V": 9.22,
    "Cr": 10.44,
    "Mn": 29.81,
    "Fe": 8.98,
    "Co": 19.83,
    "Ni": 11.21,
    "Cu": 4.81,
}
STORED_E2E_UNIVERSAL_TEST_ETA = {
    "Ti": 10.2135,
    "V": 10.4479,
    "Cr": 15.0177,
    "Mn": 23.2277,
    "Fe": 13.5697,
    "Co": 22.5677,
    "Ni": 15.0454,
    "Cu": 6.1412,
}
STORED_E2E_TUNED_TEST_ETA = {
    "Ti": 10.6063,
    "V": 11.1439,
    "Cr": 17.3720,
    "Mn": 30.5619,
    "Fe": 14.2878,
    "Co": 25.1017,
    "Ni": 15.2779,
    "Cu": 6.7511,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Prepared AnionXAS directory. Uses ANIONXAS_UNIVERSAL_DATA_DIR "
            "or output/anionxas_universal_prepared."
        ),
    )
    parser.add_argument(
        "--training-root",
        default=None,
        help=(
            "Training output root. Uses ANIONXAS_TRAINING_ROOT or "
            "output/training/anionxasUniversal."
        ),
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Name of this training run.")
    parser.add_argument(
        "--skip-universal",
        action="store_true",
        help="Do not train UniversalXAS. Reuse its expected existing checkpoint.",
    )
    parser.add_argument(
        "--run-tuned",
        action="store_true",
        help="Train or reuse the optional tuned UniversalXAS heads.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing best checkpoints and retrain requested heads.",
    )
    parser.add_argument(
        "--tuned-elements",
        nargs="+",
        default=DEFAULT_TUNED_ELEMENTS,
        metavar="ELEMENT",
        help="Tuned head elements, or exactly one 'all' to use every dataset element.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES value. If omitted, Lightning selects the device.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize data, then skip outputs and checkpoint work.",
    )
    return parser.parse_args()


def resolve_path(value: str | None, environment_name: str, fallback: Path) -> Path:
    """Resolve a CLI path, environment path, or notebook-compatible fallback."""
    raw_value = value if value is not None else os.environ.get(environment_name, fallback)
    return Path(raw_value).expanduser().resolve()


def load_and_validate_dataset(data_dir: Path):
    """Load arrays and enforce the notebook's data invariants."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Prepared AnionXAS directory does not exist: {data_dir}")

    X = np.load(data_dir / "X.npy", allow_pickle=False)
    y = np.load(data_dir / "y.npy", allow_pickle=False)
    elements = np.load(data_dir / "elements.npy", allow_pickle=False).astype(str)
    material_ids = np.load(data_dir / "material_ids.npy", allow_pickle=False).astype(str)
    sites = np.load(data_dir / "sites.npy", allow_pickle=False)
    split_codes = np.load(data_dir / "split_codes.npy", allow_pickle=False)
    with (data_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")

    if X.ndim != 2 or y.ndim != 2:
        raise ValueError(f"X and y must be two-dimensional; got {X.shape=} and {y.shape=}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y row counts differ: {X.shape[0]} != {y.shape[0]}")
    if X.shape[1] != INPUT_DIM or y.shape[1] != OUTPUT_DIM:
        raise ValueError(
            f"Expected X.shape[1]={INPUT_DIM} and y.shape[1]={OUTPUT_DIM}; "
            f"got {X.shape[1]=}, {y.shape[1]=}"
        )
    if X.shape[0] == 0:
        raise ValueError("The prepared dataset must contain at least one row")
    if split_codes.ndim != 1 or not np.issubdtype(split_codes.dtype, np.integer):
        raise ValueError("split_codes must be a one-dimensional integer array")
    for name, values in (
        ("elements", elements),
        ("material_ids", material_ids),
        ("sites", sites),
        ("split_codes", split_codes),
    ):
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != len(X):
            raise ValueError(f"{name} has {len(values)} rows; X has {len(X)}")
    if not np.isin(split_codes, [0, 1, 2]).all():
        raise ValueError("split_codes must contain only 0 (train), 1 (val), and 2 (test)")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")

    material_splits = {}
    for material_id, split_code in zip(material_ids, split_codes):
        material_splits.setdefault(material_id, set()).add(int(split_code))
    if any(len(codes) != 1 for codes in material_splits.values()):
        raise ValueError("Material-level split invariant failed: one material occurs in multiple splits")

    dataset_elements = sorted(np.unique(elements).tolist())
    split_counts = pd.crosstab(
        pd.Series(elements, name="element"), pd.Series(split_codes, name="split_code")
    ).reindex(index=dataset_elements, columns=[0, 1, 2], fill_value=0)
    split_counts.columns = ["train", "val", "test"]
    if (split_counts == 0).any().any():
        raise ValueError("Every dataset element needs at least one train, validation, and test row")
    return X, y, elements, material_ids, sites, split_codes, metadata, dataset_elements, split_counts


def print_dataset_summary(
    data_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    material_ids: np.ndarray,
    metadata: dict[str, Any],
    split_codes: np.ndarray,
    dataset_elements: list[str],
    split_counts: pd.DataFrame,
) -> None:
    """Print the notebook summary in terminal-readable form."""
    summary = pd.DataFrame(
        {
            "rows": [len(X)],
            "elements": [len(dataset_elements)],
            "materials": [len(np.unique(material_ids))],
            "feature_dim": [X.shape[1]],
            "spectrum_dim": [y.shape[1]],
        }
    )
    print(f"Repository: {REPO_ROOT}")
    print(f"Data:       {data_dir}")
    print(summary.to_string(index=False))
    print("Per-element split counts:")
    print(split_counts.to_string())
    print(f"Metadata keys: {sorted(metadata)}")
    print(f"Split totals: {Counter(split_codes.tolist())}")


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    *,
    sampler=None,
    shuffle: bool = False,
    training: bool = False,
) -> DataLoader:
    """Create a tensor loader with a BatchNorm-safe training batch size."""
    indices = np.flatnonzero(mask)
    if training and len(indices) < 2:
        raise ValueError(f"A training selection needs at least two rows; got {len(indices)}")
    batch_size = int(batch_size)
    if training:
        batch_size = min(batch_size, len(indices))
        # BatchNorm1d cannot process a one-sample final batch.
        while len(indices) % batch_size == 1 and batch_size > 2:
            batch_size -= 1
    dataset = TensorDataset(
        torch.as_tensor(X[indices], dtype=torch.float32),
        torch.as_tensor(y[indices], dtype=torch.float32),
    )
    loader_kwargs = {"batch_size": batch_size, "num_workers": 0}
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **loader_kwargs)


def make_module(hparams: dict[str, Any]) -> PlModule:
    """Build the notebook's XASBlock and PlModule."""
    if hparams["scheduler"] != "cosine":
        raise ValueError(f"Unsupported scheduler: {hparams['scheduler']!r}; expected 'cosine'")
    if hparams["cosine_t"] < 1:
        raise ValueError(f"cosine_t must be positive, got {hparams['cosine_t']}")
    XASBlock.DROPOUT = float(hparams["dropout"])
    return PlModule(
        XASBlock(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM),
        lr=float(hparams["lr"]),
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs={"T_max": int(hparams["cosine_t"]), "eta_min": 1e-6},
        lr_scheduler_interval="epoch",
    )


def checkpoint_state_dict(checkpoint: Path) -> dict[str, Any]:
    """Load a Lightning state dictionary and fail on malformed checkpoints."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    try:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint: {checkpoint}") from exc
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint}")
    return state_dict


def fit_head(
    module: PlModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    hparams: dict[str, Any],
    *,
    force_retrain: bool = False,
    initial_checkpoint: Path | None = None,
) -> Path:
    """Reuse or train one head and return its stable best-checkpoint path."""
    best_path = run_dir / "best.ckpt"
    if force_retrain and run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f"Training run path is not a directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if best_path.exists() and not force_retrain:
        module.load_state_dict(checkpoint_state_dict(best_path), strict=True)
        return best_path
    if initial_checkpoint is not None:
        module.load_state_dict(checkpoint_state_dict(initial_checkpoint), strict=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor=hparams["monitor"],
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor=hparams["monitor"], mode="min", patience=int(hparams["patience"])
    )
    csv_logger = CSVLogger(save_dir=str(run_dir), name="csv_logs", version="0")
    trainer = Trainer(
        max_epochs=int(hparams["max_epochs"]),
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=csv_logger,
        default_root_dir=str(run_dir),
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not checkpoint_callback.best_model_path:
        raise RuntimeError(f"No best checkpoint was written in {run_dir}")
    produced_path = Path(checkpoint_callback.best_model_path)
    if produced_path != best_path:
        shutil.copy2(produced_path, best_path)
    return best_path


def predict_checkpoint(checkpoint: Path, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Predict from one checkpoint on CPU in deterministic batches."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        raise ValueError(f"Prediction X must have shape (n, {INPUT_DIM}); got {features.shape}")
    if len(features) == 0:
        return np.empty((0, OUTPUT_DIM), dtype=np.float32)
    XASBlock.DROPOUT = 0.0
    module = PlModule(XASBlock(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM), lr=1e-4)
    module.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    module.eval()
    loader = DataLoader(torch.as_tensor(features), batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            predictions.append(module.model(batch).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    if prediction.shape != (len(features), OUTPUT_DIM) or not np.isfinite(prediction).all():
        raise ValueError(f"Malformed checkpoint prediction shape or values: {prediction.shape}")
    return prediction


def eta_from_predictions(
    y_train: np.ndarray, y_eval: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    """Return eta and model median per-spectrum MSE."""
    y_train = np.asarray(y_train, dtype=np.float64)
    y_eval = np.asarray(y_eval, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(y_train) == 0 or len(y_eval) == 0:
        raise ValueError("eta requires non-empty train and evaluation targets")
    if y_eval.shape != predictions.shape:
        raise ValueError(f"Evaluation target and prediction shapes differ: {y_eval.shape} != {predictions.shape}")
    train_mean = y_train.mean(axis=0)
    baseline_mse = np.mean((y_eval - train_mean) ** 2, axis=1)
    model_mse = np.mean((y_eval - predictions) ** 2, axis=1)
    baseline_median = float(np.median(baseline_mse))
    model_median = float(np.median(model_mse))
    if not np.isfinite(model_median) or model_median <= 0:
        raise ZeroDivisionError(
            "Model median MSE must be finite and positive for eta; "
            f"got {model_median}"
        )
    return baseline_median / model_median, model_median


def mask_for_element(
    element: str,
    split_code: int,
    elements: np.ndarray,
    split_codes: np.ndarray,
    dataset_elements: list[str],
) -> np.ndarray:
    if element not in dataset_elements:
        raise ValueError(f"Unknown element: {element}")
    return (elements == element) & (split_codes == split_code)


def resolve_tuned_elements(requested: list[str], dataset_elements: list[str]) -> list[str]:
    if not all(isinstance(element, str) for element in requested):
        raise ValueError("--tuned-elements must contain element names as strings")
    all_requested = any(element.strip().lower() == "all" for element in requested)
    if all_requested:
        if len(requested) != 1:
            raise ValueError("--tuned-elements='all' cannot be mixed with named elements")
        return list(dataset_elements)
    unknown = sorted(set(requested) - set(dataset_elements))
    if unknown:
        raise ValueError(f"Unknown configured element name(s): {unknown}")
    return list(requested)


def universal_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "heads" / "universalXAS" / "All_elements" / "runs" / "e2e_head_hparams" / "best.ckpt"


def collect_tuned_checkpoints(
    *,
    run_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    elements: np.ndarray,
    split_codes: np.ndarray,
    dataset_elements: list[str],
    tuned_elements: list[str],
    run_tuned: bool,
    force_retrain: bool,
    universal_checkpoint: Path | None,
) -> dict[tuple[str, str], Path]:
    """Train requested tuned heads or discover prior tuned heads."""
    tuned_root = run_dir / "heads" / "tunedUniversalXAS"
    tuned_checkpoints = {}
    if run_tuned and universal_checkpoint is None:
        raise FileNotFoundError("--run-tuned requires the universal best checkpoint")

    for element in tuned_elements:
        train_mask = mask_for_element(element, 0, elements, split_codes, dataset_elements)
        val_mask = mask_for_element(element, 1, elements, split_codes, dataset_elements)
        for setting in TUNED_SETTINGS:
            setting_name = str(setting["name"])
            setting_dir = tuned_root / element / "runs" / setting_name
            checkpoint = setting_dir / "best.ckpt"
            if checkpoint.exists() and not checkpoint.is_file():
                raise ValueError(f"Tuned checkpoint path is not a file: {checkpoint}")
            if run_tuned:
                setting_hparams = {**setting, "monitor": "val_median_mse"}
                tuned_module = make_module(setting_hparams)
                tuned_train_loader = make_loader(
                    X, y, train_mask, setting["batch_size"], shuffle=True, training=True
                )
                tuned_val_loader = make_loader(X, y, val_mask, setting["batch_size"])
                checkpoint = fit_head(
                    tuned_module,
                    tuned_train_loader,
                    tuned_val_loader,
                    setting_dir,
                    setting_hparams,
                    force_retrain=force_retrain,
                    initial_checkpoint=universal_checkpoint,
                )
            if checkpoint.is_file():
                tuned_checkpoints[(element, setting_name)] = checkpoint

    if not run_tuned:
        # Discover prior runs for every valid dataset element, not only configured elements.
        for checkpoint in sorted(tuned_root.glob("*/runs/*/best.ckpt")):
            element = checkpoint.parents[2].name
            if element not in dataset_elements:
                raise ValueError(f"Found tuned checkpoint for unknown element: {checkpoint}")
            tuned_checkpoints[(element, checkpoint.parent.name)] = checkpoint
    return tuned_checkpoints


def evaluate_and_save(
    *,
    run_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    elements: np.ndarray,
    split_codes: np.ndarray,
    dataset_elements: list[str],
    universal_checkpoint: Path | None,
    tuned_checkpoints: dict[tuple[str, str], Path],
) -> None:
    """Evaluate checkpoints and write the notebook's four CSV and PNG outputs."""
    tuned_validation_rows = []
    for (element, setting_name), checkpoint in tuned_checkpoints.items():
        element_train = mask_for_element(element, 0, elements, split_codes, dataset_elements)
        element_val = mask_for_element(element, 1, elements, split_codes, dataset_elements)
        val_eta, val_median_mse = eta_from_predictions(
            y[element_train], y[element_val], predict_checkpoint(checkpoint, X[element_val])
        )
        tuned_validation_rows.append(
            [
                element,
                setting_name,
                str(checkpoint),
                val_eta,
                val_median_mse,
                int(element_train.sum()),
                int(element_val.sum()),
            ]
        )
    tuned_validation_columns = [
        "element",
        "setting",
        "checkpoint",
        "val_eta",
        "val_median_mse",
        "n_train",
        "n_val",
    ]
    tuned_validation_candidates = pd.DataFrame(
        tuned_validation_rows, columns=tuned_validation_columns
    )
    tuned_validation_candidates.to_csv(run_dir / "tuned_validation_candidates.csv", index=False)
    if tuned_validation_candidates.empty:
        selected_tuned = tuned_validation_candidates.copy()
        print("No tuned checkpoints were trained or found.")
    else:
        selected_tuned = (
            tuned_validation_candidates.sort_values(
                ["element", "val_eta"], ascending=[True, False]
            )
            .drop_duplicates("element", keep="first")
            .reset_index(drop=True)
        )
        print("Selected tuned heads:")
        print(selected_tuned.to_string(index=False))

    universal_columns = [
        "element",
        "n_train",
        "n_val",
        "n_test",
        "val_eta",
        "test_eta",
        "val_median_mse",
        "test_median_mse",
    ]
    universal_rows = []
    if universal_checkpoint is not None:
        for element in dataset_elements:
            element_train = mask_for_element(element, 0, elements, split_codes, dataset_elements)
            element_val = mask_for_element(element, 1, elements, split_codes, dataset_elements)
            element_test = mask_for_element(element, 2, elements, split_codes, dataset_elements)
            val_eta, val_median_mse = eta_from_predictions(
                y[element_train], y[element_val], predict_checkpoint(universal_checkpoint, X[element_val])
            )
            test_eta, test_median_mse = eta_from_predictions(
                y[element_train], y[element_test], predict_checkpoint(universal_checkpoint, X[element_test])
            )
            universal_rows.append(
                {
                    "element": element,
                    "n_train": int(element_train.sum()),
                    "n_val": int(element_val.sum()),
                    "n_test": int(element_test.sum()),
                    "val_eta": val_eta,
                    "test_eta": test_eta,
                    "val_median_mse": val_median_mse,
                    "test_median_mse": test_median_mse,
                }
            )
    universal_eval = pd.DataFrame(universal_rows, columns=universal_columns).sort_values(
        "element"
    ).reset_index(drop=True)
    universal_eval.to_csv(run_dir / "universal_eval_by_element.csv", index=False)
    if universal_eval.empty:
        print("No universal checkpoint is available for evaluation.")
    else:
        print("Universal evaluation:")
        print(universal_eval.to_string(index=False))

    tuned_columns = [
        "element",
        "setting",
        "checkpoint",
        "n_train",
        "n_val",
        "n_test",
        "val_eta",
        "test_eta",
        "val_median_mse",
        "test_median_mse",
    ]
    tuned_eval_rows = []
    for record in selected_tuned.to_dict("records"):
        element = record["element"]
        element_train = mask_for_element(element, 0, elements, split_codes, dataset_elements)
        element_test = mask_for_element(element, 2, elements, split_codes, dataset_elements)
        test_eta, test_median_mse = eta_from_predictions(
            y[element_train],
            y[element_test],
            predict_checkpoint(record["checkpoint"], X[element_test]),
        )
        tuned_eval_rows.append(
            [
                element,
                record["setting"],
                record["checkpoint"],
                record["n_train"],
                record["n_val"],
                int(element_test.sum()),
                record["val_eta"],
                test_eta,
                record["val_median_mse"],
                test_median_mse,
            ]
        )
    tuned_eval = pd.DataFrame(tuned_eval_rows, columns=tuned_columns).sort_values(
        "element"
    ).reset_index(drop=True)
    tuned_eval.to_csv(run_dir / "tuned_eval_by_element.csv", index=False)
    if not tuned_eval.empty:
        print("Tuned evaluation:")
        print(tuned_eval.to_string(index=False))

    universal_values = universal_eval[["element", "val_eta", "test_eta"]].rename(
        columns={
            "val_eta": "notebook_universal_val_eta",
            "test_eta": "notebook_universal_test_eta",
        }
    )
    tuned_values = tuned_eval[["element", "val_eta", "test_eta"]].rename(
        columns={
            "val_eta": "notebook_tuned_val_eta",
            "test_eta": "notebook_tuned_test_eta",
        }
    )
    comparison = (
        pd.DataFrame({"element": OMNIXAS_ELEMENTS})
        .merge(universal_values, on="element", how="left")
        .merge(tuned_values, on="element", how="left")
    )
    comparison["paper_universal_test_eta"] = comparison["element"].map(PAPER_UNIVERSAL_TEST_ETA)
    comparison["paper_tuned_test_eta"] = comparison["element"].map(PAPER_TUNED_TEST_ETA)
    comparison["stored_e2e_universal_test_eta"] = comparison["element"].map(
        STORED_E2E_UNIVERSAL_TEST_ETA
    )
    comparison["stored_e2e_tuned_test_eta"] = comparison["element"].map(
        STORED_E2E_TUNED_TEST_ETA
    )
    comparison.to_csv(run_dir / "omnixas_8_comparison.csv", index=False)
    print("OmniXAS eight-element comparison:")
    print(comparison.to_string(index=False))

    eta_columns = [
        "notebook_universal_val_eta",
        "notebook_universal_test_eta",
        "notebook_tuned_val_eta",
        "notebook_tuned_test_eta",
        "paper_universal_test_eta",
        "paper_tuned_test_eta",
        "stored_e2e_universal_test_eta",
        "stored_e2e_tuned_test_eta",
    ]
    fig, ax = plt.subplots(figsize=(18, 7))
    comparison.set_index("element")[eta_columns].plot(kind="bar", ax=ax)
    ax.set_title("datasets differ; AnionXAS 200D first benchmark")
    ax.set_xlabel("Element")
    ax.set_ylabel("eta")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(run_dir / "omnixas_8_eta_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if universal_checkpoint is not None:
        assert len(universal_eval) == len(dataset_elements), (
            "Expected one universal evaluation row per element: "
            f"{len(universal_eval)} != {len(dataset_elements)}"
        )
    assert len(comparison) == 8, f"Expected 8 OmniXAS comparison rows, got {len(comparison)}"
    print("Self-check passed.")
    print(f"Output directory: {run_dir}")


def main() -> int:
    args = parse_args()
    data_dir = resolve_path(
        args.data_dir,
        "ANIONXAS_UNIVERSAL_DATA_DIR",
        REPO_ROOT / "output" / "anionxas_universal_prepared",
    )
    training_root = resolve_path(
        args.training_root,
        "ANIONXAS_TRAINING_ROOT",
        REPO_ROOT / "output" / "training" / "anionxasUniversal",
    )
    if not args.run_name:
        raise ValueError("--run-name must not be empty")

    (
        X,
        y,
        elements,
        material_ids,
        sites,
        split_codes,
        metadata,
        dataset_elements,
        split_counts,
    ) = load_and_validate_dataset(data_dir)
    print_dataset_summary(
        data_dir,
        X,
        y,
        material_ids,
        metadata,
        split_codes,
        dataset_elements,
        split_counts,
    )
    if args.validate_only:
        return 0

    run_dir = training_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED, workers=True)
    universal_best = universal_checkpoint_path(run_dir)
    if args.skip_universal:
        if universal_best.exists() and not universal_best.is_file():
            raise ValueError(f"Universal checkpoint path is not a file: {universal_best}")
        universal_checkpoint = universal_best if universal_best.is_file() else None
    else:
        train_mask = split_codes == 0
        val_mask = split_codes == 1
        train_elements = elements[train_mask]
        element_counts = Counter(train_elements.tolist())
        sample_weights = np.asarray(
            [1.0 / element_counts[element] for element in train_elements], dtype=np.float64
        )
        universal_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        universal_train_loader = make_loader(
            X,
            y,
            train_mask,
            UNIVERSAL_HPARAMS["batch_size"],
            sampler=universal_sampler,
            training=True,
        )
        universal_val_loader = make_loader(X, y, val_mask, UNIVERSAL_HPARAMS["batch_size"])
        universal_module = make_module(UNIVERSAL_HPARAMS)
        universal_checkpoint = fit_head(
            universal_module,
            universal_train_loader,
            universal_val_loader,
            universal_best.parent,
            UNIVERSAL_HPARAMS,
            force_retrain=args.force_retrain,
        )
    if universal_checkpoint is None:
        print(
            "No universal checkpoint found; set --skip-universal off or "
            "provide the expected checkpoint."
        )
    else:
        print(f"Universal checkpoint: {universal_checkpoint}")

    tuned_elements = resolve_tuned_elements(args.tuned_elements, dataset_elements)
    tuned_checkpoints = collect_tuned_checkpoints(
        run_dir=run_dir,
        X=X,
        y=y,
        elements=elements,
        split_codes=split_codes,
        dataset_elements=dataset_elements,
        tuned_elements=tuned_elements,
        run_tuned=args.run_tuned,
        force_retrain=args.force_retrain,
        universal_checkpoint=universal_checkpoint,
    )
    evaluate_and_save(
        run_dir=run_dir,
        X=X,
        y=y,
        elements=elements,
        split_codes=split_codes,
        dataset_elements=dataset_elements,
        universal_checkpoint=universal_checkpoint,
        tuned_checkpoints=tuned_checkpoints,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
