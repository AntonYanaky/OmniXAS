import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_anionxas_universal import PreparationError, prepare_dataset


ROWS = {
    "('Cu', 'mat-b', 1)": 0,
    "('Fe', 'mat-a', 0)": 1,
    "('Cu', 'mat-a', 1)": 2,
    "('Ni', 'mat-c', 0)": 3,
    "('Fe', 'mat-b', 2)": 4,
    "('Cu', 'mat-d', 0)": 5,
}


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    feature_path = tmp_path / "features.npz"
    spectral_path = tmp_path / "spectra.npz"
    np.savez(
        feature_path,
        **{
            key: np.full(64, index, dtype=np.float64)
            for key, index in ROWS.items()
        },
    )
    np.savez(
        spectral_path,
        **{
            key: np.full(200, index, dtype=np.float64)
            for key, index in ROWS.items()
        },
    )
    return feature_path, spectral_path


def test_prepare_dataset_writes_material_level_outputs(tmp_path: Path) -> None:
    feature_path, spectral_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "prepared"

    metadata = prepare_dataset(feature_path, spectral_path, output_dir, seed=42)

    expected_files = {
        "X.npy",
        "y.npy",
        "elements.npy",
        "material_ids.npy",
        "sites.npy",
        "split_codes.npy",
        "keys.txt",
        "manifest.csv",
        "element_counts.csv",
        "split_counts_by_element.csv",
        "metadata.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files

    X = np.load(output_dir / "X.npy", allow_pickle=False)
    y = np.load(output_dir / "y.npy", allow_pickle=False)
    elements = np.load(output_dir / "elements.npy", allow_pickle=False)
    material_ids = np.load(output_dir / "material_ids.npy", allow_pickle=False)
    sites = np.load(output_dir / "sites.npy", allow_pickle=False)
    split_codes = np.load(output_dir / "split_codes.npy", allow_pickle=False)

    assert X.shape == (len(ROWS), 64)
    assert y.shape == (len(ROWS), 200)
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    assert X.flags.c_contiguous
    assert y.flags.c_contiguous
    assert sites.shape == (len(ROWS),)
    assert sites.dtype == np.int64
    assert split_codes.shape == (len(ROWS),)
    assert split_codes.dtype == np.int8
    assert len(elements) == len(material_ids) == len(sites) == len(split_codes) == len(X)

    material_splits: dict[str, set[int]] = {}
    for material_id, split_code in zip(material_ids, split_codes):
        material_splits.setdefault(str(material_id), set()).add(int(split_code))
    assert all(len(splits) == 1 for splits in material_splits.values())
    assert set(split_codes.tolist()) == {0, 1, 2}

    with (output_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == X.shape[0]
    assert [int(row["row_index"]) for row in manifest_rows] == list(range(X.shape[0]))
    assert {row["split"] for row in manifest_rows} == {"train", "val", "test"}

    metadata_from_file = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata_from_file == metadata
    assert metadata_from_file["row_count"] == len(ROWS)
    assert metadata_from_file["feature_dim"] == 64
    assert metadata_from_file["target_dim"] == 200
    assert metadata_from_file["elements"] == ["Cu", "Fe", "Ni"]
    assert sum(metadata_from_file["split_counts"].values()) == X.shape[0]


def test_prepare_dataset_rejects_wrong_dimensions(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.npz"
    spectral_path = tmp_path / "spectra.npz"
    key = "('Cu', 'mat-a', 0)"
    np.savez(feature_path, **{key: np.zeros(63, dtype=np.float32)})
    np.savez(spectral_path, **{key: np.zeros(200, dtype=np.float32)})

    with pytest.raises(PreparationError, match="length 64"):
        prepare_dataset(feature_path, spectral_path, tmp_path / "prepared")
