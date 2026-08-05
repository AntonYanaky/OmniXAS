#!/usr/bin/env python3
"""Train and evaluate balanced AnionXAS UniversalXAS heads."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import uuid

# Set the device before importing torch or Lightning.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from omnixas.model.training import PlModule
from omnixas.model.xasblock import XASBlock

INPUT_DIM = 64
OUTPUT_DIM = 200
HIDDEN_DIMS = [500, 500, 550]
SEED = 42
HEAD_OBJECTIVE = "balanced_relative_mse"
OMNIXAS_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
SCOPE_NAMES = {"all": "All_elements", "omnixas8": "OmniXAS_8"}
DATA_FILES = (
    "X.npy", "y.npy", "elements.npy", "material_ids.npy", "sites.npy",
    "split_codes.npy", "metadata.json"
)
DEFAULT_DATA_DIR = Path.home() / "data" / "anionxas_drive" / "drive-download-20260804T124325Z-1-001"
DEFAULT_FEATURE_NPZ = DEFAULT_DATA_DIR / "final_feature_data.npz"
DEFAULT_SPECTRAL_NPZ = DEFAULT_DATA_DIR / "final_spectral_data.npz"
DEFAULT_RUN_NAME = "anionxas_universal_first_benchmark"
DEFAULT_TUNED_ELEMENTS = OMNIXAS_ELEMENTS.copy()
TUNED_SETTINGS = [
    {"name": "e2e_head_hparams", "lr": 7e-4, "dropout": 0.25, "max_epochs": 300,
     "patience": 30, "scheduler": "cosine", "cosine_t": 300, "batch_size": 32},
    {"name": "cosT500_lr3e-4_do0p1", "lr": 3e-4, "dropout": 0.1, "max_epochs": 1000,
     "patience": 25, "scheduler": "cosine", "cosine_t": 500, "batch_size": 32},
]
UNIVERSAL_HPARAMS = {**TUNED_SETTINGS[0], "monitor": "val_balanced_rel_mse"}
CANDIDATE_COLUMNS = [
    "element", "setting", "checkpoint", "val_eta", "val_median_mse", "n_train", "n_val"
]
UNIVERSAL_COLUMNS = [
    "element", "n_train", "n_val", "n_test", "val_eta", "test_eta",
    "val_median_mse", "test_median_mse"
]
TUNED_COLUMNS = [
    "element", "setting", "checkpoint", "n_train", "n_val", "n_test",
    "val_eta", "test_eta", "val_median_mse", "test_median_mse"
]
SPLIT_NAMES = ("train", "val", "test")
SPLIT_CODES = {name: code for code, name in enumerate(SPLIT_NAMES)}


class PreparationError(ValueError):
    """Raised when source data or preparation invariants are invalid."""


def prepare_dataset(
    feature_npz: str | os.PathLike[str],
    spectral_npz: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    seed: int = SEED,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare aligned AnionXAS arrays with deterministic material splits."""
    feature_path = Path(feature_npz).expanduser()
    spectral_path = Path(spectral_npz).expanduser()
    output_dir = Path(out_dir).expanduser()
    for path, label in ((feature_path, "feature NPZ"), (spectral_path, "spectral NPZ")):
        if not path.exists():
            raise PreparationError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise PreparationError(f"{label} is not a file: {path}")
    if output_dir.is_symlink():
        raise PreparationError(f"output path must not be a symlink: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise PreparationError(f"output path is not a directory: {output_dir}")
        if not overwrite:
            raise PreparationError(
                f"output directory already exists: {output_dir}; use --overwrite to replace it"
            )
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise PreparationError(f"output parent is not a directory: {output_dir.parent}")

    if isinstance(seed, bool):
        raise PreparationError("seed must be a non-negative integer")
    try:
        seed_int = operator.index(seed)
    except TypeError as exc:
        raise PreparationError("seed must be a non-negative integer") from exc
    if seed_int < 0:
        raise PreparationError("seed must be a non-negative integer")
    seed_int = int(seed_int)

    requested_fractions = (train_frac, val_frac, test_frac)
    checked_fractions = []
    for name, value in zip(SPLIT_NAMES, requested_fractions):
        if isinstance(value, bool):
            raise PreparationError(f"{name}_frac must be a finite non-negative number")
        try:
            fraction = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PreparationError(
                f"{name}_frac must be a finite non-negative number"
            ) from exc
        if not math.isfinite(fraction) or fraction < 0:
            raise PreparationError(f"{name}_frac must be a finite non-negative number")
        checked_fractions.append(fraction)
    fraction_total = sum(checked_fractions)
    if not math.isclose(fraction_total, 1.0, rel_tol=1e-7, abs_tol=1e-6):
        raise PreparationError(
            "train_frac + val_frac + test_frac must sum approximately to 1 "
            f"(got {fraction_total!r})"
        )
    if fraction_total == 0:
        raise PreparationError("split fractions must have a positive sum")
    normalized_fractions = tuple(fraction / fraction_total for fraction in checked_fractions)

    try:
        feature_archive = np.load(feature_path, allow_pickle=False)
    except Exception as exc:
        raise PreparationError(f"could not load feature NPZ {feature_path}: {exc}") from exc
    try:
        spectral_archive = np.load(spectral_path, allow_pickle=False)
    except Exception as exc:
        if isinstance(feature_archive, np.lib.npyio.NpzFile):
            feature_archive.close()
        raise PreparationError(f"could not load spectral NPZ {spectral_path}: {exc}") from exc

    try:
        if not isinstance(feature_archive, np.lib.npyio.NpzFile):
            raise PreparationError(f"feature input is not an NPZ archive: {feature_path}")
        if not isinstance(spectral_archive, np.lib.npyio.NpzFile):
            raise PreparationError(f"spectral input is not an NPZ archive: {spectral_path}")
        feature_keys = set(feature_archive.files)
        spectral_keys = set(spectral_archive.files)
        if feature_keys != spectral_keys:
            missing_from_spectral = sorted(feature_keys - spectral_keys)
            missing_from_feature = sorted(spectral_keys - feature_keys)
            raise PreparationError(
                "feature and spectral NPZ key sets differ; "
                f"missing from spectral={missing_from_spectral!r}, "
                f"missing from feature={missing_from_feature!r}"
            )
        if not feature_keys:
            raise PreparationError("feature and spectral NPZ archives contain no arrays")

        records = []
        for key in feature_keys:
            if "\n" in key or "\r" in key:
                raise PreparationError(
                    f"key contains a line break and cannot be written safely: {key!r}"
                )
            try:
                parsed = ast.literal_eval(key)
            except (SyntaxError, ValueError, TypeError) as exc:
                raise PreparationError(
                    f"key {key!r} is not a valid Python tuple/list literal"
                ) from exc
            if not isinstance(parsed, (tuple, list)) or len(parsed) != 3:
                raise PreparationError(
                    f"key {key!r} must parse to a 3-tuple or 3-item list"
                )
            element, material_id, raw_site = parsed
            if not isinstance(element, str):
                raise PreparationError(f"key {key!r} element must be a string")
            if not isinstance(material_id, str):
                raise PreparationError(f"key {key!r} material_id must be a string")
            if isinstance(raw_site, bool):
                raise PreparationError(
                    f"key {key!r} has a boolean site; expected an integer"
                )
            try:
                site = int(raw_site)
            except (TypeError, ValueError, OverflowError) as exc:
                raise PreparationError(
                    f"key {key!r} has a site that is not int-compatible: {raw_site!r}"
                ) from exc
            if isinstance(raw_site, float) and (
                not math.isfinite(raw_site) or not raw_site.is_integer()
            ):
                raise PreparationError(
                    f"key {key!r} has a non-integral site: {raw_site!r}"
                )
            if site < np.iinfo(np.int64).min or site > np.iinfo(np.int64).max:
                raise PreparationError(f"key {key!r} site is outside the int64 range")
            records.append((element, material_id, site, key))
        records.sort(key=lambda record: (record[0], record[1], record[2], record[3]))

        feature_rows = []
        target_rows = []
        elements = []
        material_ids = []
        sites = []
        keys = []
        for element, material_id, site, key in records:
            arrays = []
            for label, archive, expected_dim in (
                ("feature", feature_archive, INPUT_DIM),
                ("spectral", spectral_archive, OUTPUT_DIM),
            ):
                try:
                    array = np.asarray(archive[key])
                except (KeyError, ValueError, TypeError) as exc:
                    raise PreparationError(
                        f"could not read {label} array for key {key!r}: {exc}"
                    ) from exc
                if array.ndim != 1 or array.shape[0] != expected_dim:
                    raise PreparationError(
                        f"{label} array for key {key!r} must be a one-dimensional array "
                        f"of length {expected_dim}; got shape {array.shape}"
                    )
                if array.dtype.kind not in "iuf":
                    raise PreparationError(
                        f"{label} array for key {key!r} must have a real numeric dtype; "
                        f"got {array.dtype}"
                    )
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        converted = np.asarray(array, dtype=np.float32)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise PreparationError(
                        f"{label} array for key {key!r} could not be converted to float32"
                    ) from exc
                if not np.isfinite(converted).all():
                    raise PreparationError(
                        f"{label} array for key {key!r} contains non-finite values "
                        "or overflows float32"
                    )
                arrays.append(converted)
            feature_rows.append(arrays[0])
            target_rows.append(arrays[1])
            elements.append(element)
            material_ids.append(material_id)
            sites.append(site)
            keys.append(key)
    finally:
        if isinstance(feature_archive, np.lib.npyio.NpzFile):
            feature_archive.close()
        if isinstance(spectral_archive, np.lib.npyio.NpzFile):
            spectral_archive.close()

    if not feature_rows:
        raise PreparationError("no valid rows were found")
    X = np.stack(feature_rows, axis=0).astype(np.float32, copy=False)
    y = np.stack(target_rows, axis=0).astype(np.float32, copy=False)
    material_counts = Counter(material_ids)
    materials = sorted(material_counts)
    shuffled_materials = [
        materials[int(index)] for index in np.random.default_rng(seed_int).permutation(len(materials))
    ]
    ordered_materials = sorted(
        shuffled_materials, key=lambda material: -material_counts[material]
    )
    total_rows = len(material_ids)
    targets = np.asarray(normalized_fractions, dtype=np.float64) * total_rows
    current = np.zeros(len(SPLIT_NAMES), dtype=np.int64)
    material_splits = {}
    for position, material in enumerate(ordered_materials):
        remaining_materials = len(ordered_materials) - position - 1
        empty_active = [
            split
            for split, target in enumerate(targets)
            if target > 0 and current[split] == 0
        ]
        if empty_active and remaining_materials <= len(empty_active):
            candidates = empty_active
        else:
            candidates = list(range(len(SPLIT_NAMES)))
        deficits = targets - current
        if any(deficits[split] > 0 for split in range(len(SPLIT_NAMES))):
            split = max(candidates, key=lambda candidate: deficits[candidate])
        else:
            ratios = []
            for candidate in candidates:
                target = targets[candidate]
                ratios.append(
                    math.inf if target <= 0 else current[candidate] / target
                )
            split = candidates[int(np.argmin(ratios))]
        material_splits[material] = split
        current[split] += material_counts[material]
    if len(material_splits) != len(material_counts):
        raise PreparationError("internal error: not every material received a split")

    split_codes = np.asarray(
        [material_splits[material_id] for material_id in material_ids], dtype=np.int8
    )
    material_to_splits = {}
    for material_id, split_code in zip(material_ids, split_codes):
        material_to_splits.setdefault(material_id, set()).add(int(split_code))
    overlapping_materials = sorted(
        material_id
        for material_id, splits in material_to_splits.items()
        if len(splits) != 1
    )
    if overlapping_materials:
        raise PreparationError(
            f"internal error: materials occur in multiple splits: {overlapping_materials!r}"
        )
    active_split_count = sum(fraction > 0 for fraction in normalized_fractions)
    if len(material_counts) >= active_split_count:
        empty_active_splits = [
            SPLIT_NAMES[index]
            for index, fraction in enumerate(normalized_fractions)
            if fraction > 0 and not np.any(split_codes == index)
        ]
        if empty_active_splits:
            raise PreparationError(
                "internal error: an active split is empty despite enough materials: "
                f"{empty_active_splits!r}"
            )

    split_counts_array = np.bincount(split_codes.astype(np.int64), minlength=3)
    split_counts = {
        split_name: int(split_counts_array[index])
        for index, split_name in enumerate(SPLIT_NAMES)
    }
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    element_list = sorted(set(elements))
    metadata = {
        "input_paths": {
            "feature_npz": str(feature_path.resolve()),
            "spectral_npz": str(spectral_path.resolve()),
        },
        "row_count": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "target_dim": int(y.shape[1]),
        "split_fractions": {
            split_name: float(checked_fractions[index])
            for index, split_name in enumerate(SPLIT_NAMES)
        },
        "split_counts": split_counts,
        "seed": seed_int,
        "created_at": created_at,
        "elements": element_list,
        "split_code_mapping": dict(SPLIT_CODES),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        np.save(temporary_dir / "X.npy", X)
        np.save(temporary_dir / "y.npy", y)
        np.save(temporary_dir / "elements.npy", np.asarray(elements))
        np.save(temporary_dir / "material_ids.npy", np.asarray(material_ids))
        np.save(temporary_dir / "sites.npy", np.asarray(sites, dtype=np.int64))
        np.save(temporary_dir / "split_codes.npy", split_codes.astype(np.int8, copy=False))
        (temporary_dir / "keys.txt").write_text(
            "".join(f"{key}\n" for key in keys), encoding="utf-8"
        )
        with (temporary_dir / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("row_index", "key", "element", "material_id", "site", "split"))
            for index, (key, element, material_id, site) in enumerate(
                zip(keys, elements, material_ids, sites)
            ):
                writer.writerow(
                    (index, key, element, material_id, site, SPLIT_NAMES[int(split_codes[index])])
                )
        element_counts = Counter(elements)
        with (temporary_dir / "element_counts.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("element", "count"))
            for element in sorted(element_counts):
                writer.writerow((element, element_counts[element]))
        split_element_counts = {
            split_name: {element: 0 for element in element_list}
            for split_name in SPLIT_NAMES
        }
        for element, split_code in zip(elements, split_codes):
            split_element_counts[SPLIT_NAMES[int(split_code)]][element] += 1
        with (temporary_dir / "split_counts_by_element.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("split", "element", "count"))
            for split_name in SPLIT_NAMES:
                for element in element_list:
                    writer.writerow((split_name, element, split_element_counts[split_name][element]))
        with (temporary_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

        expected_files = {
            "X.npy", "y.npy", "elements.npy", "material_ids.npy", "sites.npy",
            "split_codes.npy", "metadata.json", "keys.txt", "manifest.csv",
            "element_counts.csv", "split_counts_by_element.csv",
        }
        actual_files = {path.name for path in temporary_dir.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise PreparationError(
                f"internal error: prepared output files are {sorted(actual_files)!r}"
            )

        backup_dir = None
        try:
            if output_dir.exists():
                backup_dir = output_dir.parent / (
                    f".{output_dir.name}.backup-{uuid.uuid4().hex}"
                )
                os.replace(output_dir, backup_dir)
            os.replace(temporary_dir, output_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                try:
                    os.replace(backup_dir, output_dir)
                except OSError as restore_error:
                    raise OSError(
                        f"could not commit output and could not restore {output_dir}: "
                        f"{restore_error}"
                    ) from restore_error
            raise
        else:
            if backup_dir is not None:
                shutil.rmtree(backup_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return metadata


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    elements: np.ndarray
    material_ids: np.ndarray
    sites: np.ndarray
    split_codes: np.ndarray
    metadata: dict[str, Any]
    names: list[str]


def load_dataset(data_dir: Path) -> Dataset:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Prepared AnionXAS directory does not exist: {data_dir}")
    missing = [name for name in DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise ValueError(f"Prepared data is incomplete. Missing {missing!r} in {data_dir}")
    X, y, elements, material_ids, sites, split_codes = [
        np.load(data_dir / name, allow_pickle=False) for name in DATA_FILES[:6]
    ]
    elements, material_ids = elements.astype(str), material_ids.astype(str)
    try:
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read prepared metadata: {data_dir / 'metadata.json'}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    if X.ndim != 2 or y.ndim != 2 or X.shape[1:] != (INPUT_DIM,) or y.shape[1:] != (OUTPUT_DIM,):
        raise ValueError(
            f"Expected X shape (n, {INPUT_DIM}) and y shape (n, {OUTPUT_DIM}), "
            f"got {X.shape} and {y.shape}"
        )
    n = len(X)
    if not n or len(y) != n:
        raise ValueError(f"X and y must have the same non-zero row count, got {n} and {len(y)}")
    for name, values in zip(("elements", "material_ids", "sites"), (elements, material_ids, sites)):
        if values.ndim != 1 or len(values) != n:
            raise ValueError(f"{name} must be one-dimensional and aligned with X")
    if (
        split_codes.ndim != 1
        or len(split_codes) != n
        or not np.issubdtype(split_codes.dtype, np.integer)
    ):
        raise ValueError("split_codes must be an aligned one-dimensional integer array")
    if not np.isin(split_codes, (0, 1, 2)).all():
        raise ValueError("split_codes must contain only 0 (train), 1 (val), and 2 (test)")
    if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise ValueError("X and y must have numeric dtypes")
    if not np.issubdtype(sites.dtype, np.integer):
        raise ValueError("sites must have an integer dtype")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(sites).all():
        raise ValueError("X, y, and sites must contain only finite values")
    material_splits: dict[str, set[int]] = {}
    for material, split in zip(material_ids, split_codes):
        material_splits.setdefault(material, set()).add(int(split))
    if any(len(splits) != 1 for splits in material_splits.values()):
        raise ValueError(
            "Material-level split invariant failed: one material occurs in multiple splits"
        )
    names = sorted(np.unique(elements).tolist())
    missing = [
        element for element in names
        if any(not np.any((elements == element) & (split_codes == split)) for split in (0, 1, 2))
    ]
    if missing:
        raise ValueError(
            "Every dataset element needs train, validation, and test rows. "
            f"Missing rows for {missing!r}"
        )
    return Dataset(X, y, elements, material_ids, sites, split_codes, metadata, names)


def mask(data: Dataset, element: str, split: int) -> np.ndarray:
    return (data.elements == element) & (data.split_codes == split)


def baselines(data: Dataset, selected: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not selected:
        raise ValueError("At least one element is required for baseline calculation")
    unknown = sorted(set(selected) - set(data.names))
    if unknown:
        raise ValueError(f"Unknown elements in baseline calculation: {unknown!r}")
    if len(set(selected)) != len(selected):
        raise ValueError("Baseline elements must be unique")
    train_mse, val_mse = [], []
    for element in selected:
        train = data.y[mask(data, element, 0)].astype(np.float64)
        validation = data.y[mask(data, element, 1)].astype(np.float64)
        mean = train.mean(axis=0)
        train_mse.append(float(np.median(np.mean((train - mean) ** 2, axis=1))))
        val_mse.append(float(np.median(np.mean((validation - mean) ** 2, axis=1))))
    result = tuple(np.asarray(values, dtype=np.float32) for values in (train_mse, val_mse))
    for label, values in zip(("Training", "Validation"), result):
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{label} baseline MSE must be finite and positive: {values}")
    return result


def balanced_relative_mse(
    mse: torch.Tensor, element_indices: torch.Tensor, baselines: torch.Tensor
) -> torch.Tensor:
    if mse.ndim != 1 or element_indices.ndim != 1:
        raise ValueError("MSE and element index tensors must be one-dimensional")
    if not len(mse) or len(mse) != len(element_indices):
        raise ValueError("MSE and element index tensors must be non-empty and aligned")
    if baselines.ndim != 1 or not len(baselines):
        raise ValueError("Relative-MSE baselines must be a non-empty one-dimensional tensor")
    if not torch.isfinite(mse).all() or not torch.isfinite(baselines).all() or (baselines <= 0).any():
        raise ValueError("Relative-MSE inputs must be finite, and baselines must be positive")
    values = []
    for index, baseline in enumerate(baselines.to(mse.device)):
        selected = mse[element_indices == index]
        if not len(selected):
            raise ValueError(f"Validation has no rows for element index {index}")
        values.append(torch.quantile(selected, 0.5) / baseline)
    result = torch.stack(values).mean()
    if not torch.isfinite(result):
        raise ValueError("Balanced relative validation MSE is not finite")
    return result


class BalancedRelativeMSEModule(PlModule):
    def __init__(
        self,
        model: torch.nn.Module,
        training_baselines: np.ndarray,
        validation_baselines: np.ndarray,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        train = torch.as_tensor(training_baselines, dtype=torch.float32)
        validation = torch.as_tensor(validation_baselines, dtype=torch.float32)
        if (train.ndim, validation.ndim) != (1, 1) or train.shape != validation.shape or not len(train):
            raise ValueError("Training and validation baselines must be aligned one-dimensional arrays")
        if (
            not torch.isfinite(train).all()
            or not torch.isfinite(validation).all()
            or (train <= 0).any()
            or (validation <= 0).any()
        ):
            raise ValueError("Training and validation baselines must be finite and positive")
        self.register_buffer("training_baselines", train, persistent=False)
        self.register_buffer("validation_baselines", validation, persistent=False)
        self._val_mse: list[torch.Tensor] = []
        self._val_elements: list[torch.Tensor] = []

    def _indices(self, values: torch.Tensor) -> torch.Tensor:
        values = values.long()
        if (
            values.ndim != 1
            or not len(values)
            or values.min() < 0
            or values.max() >= len(self.training_baselines)
        ):
            raise ValueError("Batch contains invalid element indices")
        return values

    def _batch_mse(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, indices = batch
        indices = self._indices(indices)
        return torch.mean((y - self.model(x)) ** 2, dim=1), indices

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        mse, indices = self._batch_mse(batch)
        loss = (mse / self.training_baselines[indices]).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_raw_mse", mse.mean(), on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_mse, self._val_elements = [], []

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        mse, indices = self._batch_mse(batch)
        self._val_mse.append(mse.detach())
        self._val_elements.append(indices.detach())
        self.log("val_loss", mse.mean(), on_step=False, on_epoch=True)
        return mse.mean()

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if not self._val_mse:
            raise ValueError("Validation produced no batches")
        mse, indices = torch.cat(self._val_mse), torch.cat(self._val_elements)
        self.log("val_median_mse", torch.quantile(mse, 0.5), on_step=False, on_epoch=True)
        self.log(
            "val_balanced_rel_mse",
            balanced_relative_mse(mse, indices, self.validation_baselines),
            on_step=False,
            on_epoch=True,
        )


def checkpoint_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint: {path}") from exc
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state_dict: {path}")
    return state


def setting_name(hparams: dict[str, Any]) -> str:
    return f"{hparams['name']}_{HEAD_OBJECTIVE}"


def head_run_dir(run_dir: Path, family: str, scope: str, setting: str, element: str | None = None) -> Path:
    path = run_dir / "heads" / family / scope
    if element is not None:
        path /= element
    return path / "runs" / setting


def source_manifest(universal: Path) -> dict[str, str]:
    if not universal.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {universal}")
    with universal.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"universal_checkpoint": str(universal.resolve()), "universal_checkpoint_sha256": digest}


def check_manifest(setting_dir: Path, universal: Path) -> None:
    path = setting_dir / "source_universal.json"
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Tuned checkpoint provenance is missing: {path}") from exc
    if actual != source_manifest(universal):
        raise ValueError(f"Tuned checkpoint source does not match {universal}: {setting_dir}")


def train_head(
    data: Dataset,
    selected: list[str],
    hparams: dict[str, Any],
    train_base: np.ndarray,
    val_base: np.ndarray,
    run_dir: Path,
    force: bool = False,
    source: Path | None = None,
) -> Path:
    """Reuse or train one balanced head, with optional UniversalXAS transfer."""
    if hparams["scheduler"] != "cosine" or int(hparams["cosine_t"]) < 1:
        raise ValueError("Only a positive cosine scheduler with a positive period is supported")
    batch_size = int(hparams["batch_size"])
    if batch_size < 2:
        raise ValueError("Training batch size must be at least two for BatchNorm1d")
    if not selected:
        raise ValueError("At least one element is required for head training")
    best = run_dir / "best.ckpt"
    if best.exists() and not best.is_file() and not force:
        raise ValueError(f"Checkpoint path is not a file: {best}")
    if source is not None and best.is_file() and not force:
        check_manifest(run_dir, source)
    XASBlock.DROPOUT = float(hparams["dropout"])
    module = BalancedRelativeMSEModule(
        XASBlock(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM), train_base, val_base,
        lr=float(hparams["lr"]), lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs={"T_max": int(hparams["cosine_t"]), "eta_min": 1e-6},
        lr_scheduler_interval="epoch",
    )
    if best.is_file() and not force:
        module.load_state_dict(checkpoint_state(best), strict=True)
        return best
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f"Training run path is not a directory: {run_dir}")
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Incomplete training run exists without a reusable best checkpoint: "
                         f"{run_dir}. Use --force-retrain")
    run_dir.mkdir(parents=True, exist_ok=True)
    scope = np.isin(data.elements, selected)
    train_rows, val_rows = [np.flatnonzero(scope & (data.split_codes == s)) for s in (0, 1)]
    if len(train_rows) < 2:
        raise ValueError(f"Training selection needs at least two rows, got {len(train_rows)}")
    if not len(val_rows):
        raise ValueError("Validation selection is empty")
    lookup = {element: index for index, element in enumerate(selected)}
    indices = np.asarray([lookup.get(element, -1) for element in data.elements])
    train_indices, val_indices = indices[train_rows], indices[val_rows]
    if (train_indices < 0).any() or (val_indices < 0).any():
        raise ValueError(
            "Training or validation selection contains an element outside the selected head scope"
        )
    counts = np.bincount(train_indices, minlength=len(selected))
    if (counts == 0).any():
        missing = [selected[i] for i, count in enumerate(counts) if not count]
        raise ValueError(f"Training selection has no rows for element(s): {missing!r}")
    sampler = WeightedRandomSampler(
        torch.as_tensor(1.0 / counts[train_indices], dtype=torch.double), len(train_rows),
        replacement=True, generator=torch.Generator().manual_seed(SEED)
    )

    def tensor_dataset(rows: np.ndarray, indices: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.as_tensor(data.X[rows], dtype=torch.float32),
            torch.as_tensor(data.y[rows], dtype=torch.float32),
            torch.as_tensor(indices, dtype=torch.long),
        )

    train_batch = min(batch_size, len(train_rows))
    while train_batch > 2 and len(train_rows) % train_batch == 1:
        train_batch -= 1
    train_loader = DataLoader(
        tensor_dataset(train_rows, train_indices), batch_size=train_batch,
        sampler=sampler, drop_last=len(train_rows) % train_batch == 1, num_workers=0
    )
    val_loader = DataLoader(
        tensor_dataset(val_rows, val_indices), batch_size=min(batch_size, len(val_rows)),
        shuffle=False, num_workers=0
    )
    if source is not None:
        module.load_state_dict(checkpoint_state(source), strict=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir), filename="best", monitor=hparams["monitor"], mode="min",
        save_top_k=1, save_last=False, auto_insert_metric_name=False
    )
    Trainer(
        max_epochs=int(hparams["max_epochs"]), accelerator="auto", devices=1,
        callbacks=[
            checkpoint,
            EarlyStopping(
                monitor=hparams["monitor"], mode="min", patience=int(hparams["patience"])
            ),
        ],
        logger=False, default_root_dir=str(run_dir)
    ).fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError(f"No best checkpoint was written in {run_dir}")
    produced = Path(checkpoint.best_model_path)
    if produced.resolve() != best.resolve():
        shutil.copy2(produced, best)
    if source is not None:
        (run_dir / "source_universal.json").write_text(
            json.dumps(source_manifest(source), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return best


def predict(path: Path, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != INPUT_DIM:
        raise ValueError(f"Prediction X must have shape (n, {INPUT_DIM}), got {X.shape}")
    XASBlock.DROPOUT = 0.0
    module = PlModule(XASBlock(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM), lr=1e-4)
    module.load_state_dict(checkpoint_state(path), strict=True)
    module.eval()
    outputs = []
    with torch.inference_mode():
        for batch in DataLoader(
            torch.as_tensor(X, dtype=torch.float32),
            batch_size=1024,
            shuffle=False,
            num_workers=0,
        ):
            outputs.append(module.model(batch).cpu().numpy())
    prediction = np.concatenate(outputs) if outputs else np.empty((0, OUTPUT_DIM), dtype=np.float32)
    if prediction.shape != (len(X), OUTPUT_DIM) or not np.isfinite(prediction).all():
        raise ValueError(f"Malformed checkpoint prediction: {prediction.shape}")
    return prediction


def eta(data: Dataset, element: str, split: int, predictions: np.ndarray) -> tuple[float, float]:
    target = data.y[mask(data, element, split)].astype(np.float64)
    train = data.y[mask(data, element, 0)].astype(np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if not len(train) or not len(target):
        raise ValueError("eta requires non-empty training and evaluation targets")
    if predictions.shape != target.shape:
        raise ValueError("Evaluation target and prediction shapes differ: "
                         f"{target.shape} != {predictions.shape}")
    mean = train.mean(axis=0)
    baseline = float(np.median(np.mean((target - mean) ** 2, axis=1)))
    model_mse = float(np.median(np.mean((target - predictions) ** 2, axis=1)))
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"Baseline median MSE must be finite and positive, got {baseline}")
    if not np.isfinite(model_mse) or model_mse <= 0:
        raise ZeroDivisionError(f"Model median MSE must be finite and positive, got {model_mse}")
    return baseline / model_mse, model_mse


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--feature-npz", type=Path, default=DEFAULT_FEATURE_NPZ)
    parser.add_argument("--spectral-npz", type=Path, default=DEFAULT_SPECTRAL_NPZ)
    parser.add_argument("--training-root", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--universal-elements", choices=tuple(SCOPE_NAMES), default="all")
    parser.add_argument("--skip-universal", action="store_true")
    parser.add_argument("--run-tuned", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--tuned-elements", nargs="+", default=DEFAULT_TUNED_ELEMENTS, metavar="ELEMENT")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    data_value = args.data_dir if args.data_dir is not None else os.environ.get(
        "ANIONXAS_UNIVERSAL_DATA_DIR", REPO_ROOT / "output" / "anionxas_universal_prepared"
    )
    training_value = args.training_root if args.training_root is not None else os.environ.get(
        "ANIONXAS_TRAINING_ROOT", REPO_ROOT / "output" / "training" / "anionxas"
    )
    data_dir, training_root = (
        Path(data_value).expanduser().resolve(),
        Path(training_value).expanduser().resolve(),
    )
    if not args.run_name:
        raise ValueError("--run-name must not be empty")
    if not data_dir.exists():
        print(f"Preparing missing data in {data_dir}")
        prepare_dataset(
            args.feature_npz, args.spectral_npz, data_dir,
            seed=SEED, train_frac=0.8, val_frac=0.1, test_frac=0.1, overwrite=False,
        )
    elif not data_dir.is_dir():
        raise ValueError(f"Prepared data path exists but is not a directory: {data_dir}")
    data = load_dataset(data_dir)
    print(f"Data: {data_dir}")
    print(
        f"Rows: {len(data.X)}  Materials: {len(np.unique(data.material_ids))}  "
        f"Elements: {len(data.names)}  Dimensions: {data.X.shape[1]}->{data.y.shape[1]}"
    )
    print("Element split counts:")
    for element in data.names:
        counts = [int(mask(data, element, split).sum()) for split in (0, 1, 2)]
        print(f"  {element}: train={counts[0]} val={counts[1]} test={counts[2]}")
    print(f"Split totals: {dict(sorted(Counter(data.split_codes.tolist()).items()))}")
    print(f"Metadata keys: {sorted(data.metadata)}")
    if args.validate_only:
        return 0

    universal = data.names.copy() if args.universal_elements == "all" else OMNIXAS_ELEMENTS.copy()
    if args.universal_elements != "all":
        missing = sorted(set(OMNIXAS_ELEMENTS) - set(data.names))
        if missing:
            raise ValueError(
                "Universal scope 'omnixas8' requires every OmniXAS element: " f"{missing!r}"
            )
    universal_train, universal_val = baselines(data, universal)
    scope_name = SCOPE_NAMES[args.universal_elements]
    run_dir = training_root / args.run_name
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f"Training run path is not a directory: {run_dir}")
    scope_root = run_dir / "heads" / "universalXAS" / scope_name
    digest = hashlib.sha256()
    for name in ("X", "y", "elements", "material_ids", "sites", "split_codes"):
        value = np.asarray(getattr(data, name))
        header = json.dumps(
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update((value if value.flags.c_contiguous else np.ascontiguousarray(value)).view(np.uint8))
    expected = {
        "seed": SEED, "objective": HEAD_OBJECTIVE, "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM, "hidden_dims": HIDDEN_DIMS,
        "universal_scope": args.universal_elements, "universal_scope_name": scope_name,
        "universal_elements": universal, "data_dir": str(data_dir.resolve()),
        "dataset_fingerprint": digest.hexdigest(),
        "metadata": {key: value for key, value in data.metadata.items() if key != "created_at"},
        "baselines": {"training": universal_train.tolist(), "validation": universal_val.tolist()},
        "universal_hyperparameters": UNIVERSAL_HPARAMS, "tuned_hyperparameters": TUNED_SETTINGS,
        "checkpoint_layout": "heads/<family>/<scope>/runs/<setting>/best.ckpt",
    }
    if scope_root.exists() and not scope_root.is_dir():
        raise ValueError(f"Training scope path is not a directory: {scope_root}")
    config = scope_root / "training_config.json"
    if config.exists() and not config.is_file():
        raise ValueError(f"Training config path is not a file: {config}")
    if config.is_file():
        try:
            actual = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read training config: {config}") from exc
        if actual != expected and (args.skip_universal or not args.force_retrain):
            raise ValueError(
                f"Existing training config does not match requested provenance: {config}"
            )
    elif args.skip_universal:
        raise FileNotFoundError(f"Required training config does not exist: {config}")
    elif scope_root.exists() and any(scope_root.rglob("*.ckpt")) and not args.force_retrain:
        raise ValueError(f"Training provenance is missing: {config}. Use --force-retrain")
    scope_root.mkdir(parents=True, exist_ok=True)
    if not config.exists() or args.force_retrain:
        config.write_text(
            json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

    seed_everything(SEED, workers=True)
    universal_run = head_run_dir(run_dir, "universalXAS", scope_name, setting_name(UNIVERSAL_HPARAMS))
    universal_best = universal_run / "best.ckpt"
    if args.skip_universal:
        if not universal_best.is_file():
            raise FileNotFoundError(
                f"--skip-universal requires the requested checkpoint: {universal_best}"
            )
        universal_checkpoint = universal_best
    else:
        universal_checkpoint = train_head(
            data, universal, UNIVERSAL_HPARAMS, universal_train, universal_val,
            universal_run, force=args.force_retrain,
        )
    print(f"Universal checkpoint: {universal_checkpoint}")
    tuned_checkpoints: dict[tuple[str, str], Path] = {}
    tuned_root = run_dir / "heads" / "tunedUniversalXAS" / scope_name
    tuned: list[str] = []
    if args.run_tuned or tuned_root.exists():
        if any(element.lower() == "all" for element in args.tuned_elements):
            if len(args.tuned_elements) != 1:
                raise ValueError("--tuned-elements='all' cannot be mixed with named elements")
            tuned = data.names.copy()
        else:
            unknown = sorted(set(args.tuned_elements) - set(data.names))
            if unknown:
                raise ValueError(f"Unknown configured element name(s): {unknown}")
            tuned = args.tuned_elements.copy()
        outside = sorted(set(tuned) - set(universal))
        if outside:
            raise ValueError(f"Tuned elements are outside the universal scope: {outside!r}")
        if len(set(tuned)) != len(tuned):
            raise ValueError(f"Tuned elements must be unique: {tuned!r}")
    if args.run_tuned:
        for element in tuned:
            train_base, val_base = baselines(data, [element])
            for setting in TUNED_SETTINGS:
                hparams = {**setting, "monitor": "val_balanced_rel_mse"}
                name = setting_name(hparams)
                tuned_checkpoints[(element, name)] = train_head(
                    data, [element], hparams, train_base, val_base,
                    head_run_dir(run_dir, "tunedUniversalXAS", scope_name, name, element),
                    force=args.force_retrain, source=universal_checkpoint,
                )
    else:
        if tuned_root.exists() and not tuned_root.is_dir():
            raise ValueError(f"Tuned scope path is not a directory: {tuned_root}")
        for element in tuned:
            runs = tuned_root / element / "runs"
            if not runs.exists():
                continue
            if not runs.is_dir():
                raise ValueError(f"Tuned runs path is not a directory: {runs}")
            for setting_dir in sorted(path for path in runs.iterdir() if path.is_dir()):
                checkpoint = setting_dir / "best.ckpt"
                if HEAD_OBJECTIVE not in setting_dir.name or not checkpoint.exists():
                    continue
                if not checkpoint.is_file():
                    raise ValueError(f"Checkpoint path is not a file: {checkpoint}")
                check_manifest(setting_dir, universal_checkpoint)
                tuned_checkpoints[(element, setting_dir.name)] = checkpoint

    universal_key = universal_checkpoint.resolve()
    prediction_paths = {universal_key: universal_checkpoint}
    prediction_rows = {
        universal_key: np.flatnonzero(np.isin(data.elements, universal) & (data.split_codes != 0))
    }
    for (element, _), checkpoint in tuned_checkpoints.items():
        key = checkpoint.resolve()
        rows = np.flatnonzero(mask(data, element, 1) | mask(data, element, 2))
        prediction_paths.setdefault(key, checkpoint)
        if key in prediction_rows:
            prediction_rows[key] = np.union1d(prediction_rows[key], rows)
        else:
            prediction_rows[key] = rows
    cache = {
        key: (prediction_rows[key], predict(path, data.X[prediction_rows[key]]))
        for key, path in prediction_paths.items()
    }

    candidates = []
    for (element, setting), checkpoint in tuned_checkpoints.items():
        rows, predictions = cache[checkpoint.resolve()]
        validation = mask(data, element, 1)[rows]
        val_eta, val_mse = eta(data, element, 1, predictions[validation])
        candidates.append({
            "element": element, "setting": setting, "checkpoint": str(checkpoint),
            "val_eta": val_eta, "val_median_mse": val_mse,
            "n_train": int(mask(data, element, 0).sum()), "n_val": int(validation.sum()),
        })
    candidates.sort(key=lambda row: (row["element"], -row["val_eta"], row["setting"]))
    selected = {}
    for row in candidates:
        selected.setdefault(row["element"], row)
    universal_rows, universal_predictions = cache[universal_key]
    universal_results = []
    for element in universal:
        validation, test = mask(data, element, 1)[universal_rows], mask(data, element, 2)[universal_rows]
        val_eta, val_mse = eta(data, element, 1, universal_predictions[validation])
        test_eta, test_mse = eta(data, element, 2, universal_predictions[test])
        universal_results.append({
            "element": element, "n_train": int(mask(data, element, 0).sum()),
            "n_val": int(validation.sum()), "n_test": int(test.sum()),
            "val_eta": val_eta, "test_eta": test_eta,
            "val_median_mse": val_mse, "test_median_mse": test_mse,
        })
    tuned_results = []
    for row in selected.values():
        element = row["element"]
        rows, predictions = cache[Path(row["checkpoint"]).resolve()]
        test = mask(data, element, 2)[rows]
        test_eta, test_mse = eta(data, element, 2, predictions[test])
        tuned_results.append({
            **row, "n_test": int(test.sum()), "test_eta": test_eta,
            "test_median_mse": test_mse,
        })

    evaluation_dir = run_dir / "evaluations" / scope_name
    if evaluation_dir.exists() and not evaluation_dir.is_dir():
        raise ValueError(f"Evaluation output path is not a directory: {evaluation_dir}")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_csv(evaluation_dir / "tuned_validation_candidates.csv", CANDIDATE_COLUMNS, candidates)
    if selected:
        print("Selected tuned heads:")
        for row in selected.values():
            print(f"  {row['element']}: {row['setting']} (validation eta={row['val_eta']:.6g})")
    write_csv(evaluation_dir / "universal_eval_by_element.csv", UNIVERSAL_COLUMNS, universal_results)
    write_csv(evaluation_dir / "tuned_eval_by_element.csv", TUNED_COLUMNS, tuned_results)
    print(f"Output directory: {evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
