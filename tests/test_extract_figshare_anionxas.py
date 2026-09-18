import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest


_EXTRACTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "tutorial_omnixas"
    / "extract_figshare_anionxas.py"
)
_EXTRACTOR_SPEC = importlib.util.spec_from_file_location(
    "omnixas_tutorial_anionxas_extractor", _EXTRACTOR_PATH
)
assert _EXTRACTOR_SPEC is not None and _EXTRACTOR_SPEC.loader is not None
_extractor = importlib.util.module_from_spec(_EXTRACTOR_SPEC)
sys.modules[_EXTRACTOR_SPEC.name] = _extractor
_EXTRACTOR_SPEC.loader.exec_module(_extractor)

CuratedKey = _extractor.CuratedKey
ExtractionError = _extractor.ExtractionError
assign_material_splits = _extractor.assign_material_splits
extract_dataset = _extractor.extract_dataset
create_target_package = _extractor.create_target_package
load_target_package = _extractor.load_target_package
load_aligned_key_archives = _extractor.load_aligned_key_archives
load_curated_targets = _extractor.load_curated_targets
load_key_file = _extractor.load_key_file
export_clean_key_file = _extractor.export_clean_key_file
resample_target = _extractor.resample_target


def _structure(element: str, shift: float = 0.0) -> dict:
    return {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0,
        "lattice": {
            "matrix": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            "pbc": [True, True, True],
        },
        "sites": [
            {
                "species": [{"element": element, "occu": 1}],
                "abc": [shift, 0.0, 0.0],
                "xyz": [3.0 * shift, 0.0, 0.0],
                "label": element,
                "properties": {},
            }
        ],
    }


def _record(element: str, material_id: str, shift: float = 0.0) -> dict:
    energy = np.linspace(100.0, 135.0, 6)
    intensity = np.linspace(0.0, 1.0, 6)
    return {
        "absorbing_atom": 0,
        "structure": _structure(element, shift),
        "spectrum": np.column_stack((energy, intensity)).tolist(),
        "metadata": {
            "mp_id": material_id,
            "absorbing_atom_index": 0,
        },
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    key_names = [
        "('Cu', 'mp-1', 0)",
        "('Fe', 'mp-2', 0)",
        "('Ni', 'mp-3', 0)",
    ]
    feature_path = tmp_path / "clean_feature_data.npz"
    spectral_path = tmp_path / "clean_spectral_data.npz"
    np.savez(
        feature_path,
        **{key: np.full(64, index, dtype=np.float64) for index, key in enumerate(key_names)},
    )
    np.savez(
        spectral_path,
        **{
            key: np.linspace(index, index + 1, 200, dtype=np.float64)
            for index, key in enumerate(key_names)
        },
    )

    json_path = tmp_path / "xas.json"
    records = [
        _record("Cu", "mp-1"),
        _record("Fe", "mp-2"),
        _record("Ni", "mp-3"),
        _record("Cu", "mp-1", shift=0.25),
        _record("O", "mp-unselected"),
    ]
    json_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    archive_path = tmp_path / "xas.json.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(json_path, arcname="xas.json")
    return archive_path, feature_path, spectral_path, key_names


def test_resample_target_uses_141_point_relative_grid() -> None:
    source = np.linspace(0.0, 35.0, 200)
    result = resample_target(source)

    assert result.shape == (141,)
    assert result.dtype == np.float32
    assert result == pytest.approx(np.linspace(0.0, 35.0, 141))


def test_resample_target_rejects_any_negative_value() -> None:
    source = np.zeros(200)
    source[37] = -1e-12

    with pytest.raises(ExtractionError, match="negative value"):
        resample_target(source)


def test_global_split_keeps_material_together() -> None:
    keys = [
        CuratedKey("Cu", "mp-shared", 0),
        CuratedKey("Fe", "mp-shared", 1),
        CuratedKey("Cu", "mp-2", 0),
        CuratedKey("Cu", "mp-3", 0),
    ]
    assignment, row_counts = assign_material_splits(keys, seed=42)

    assert set(assignment) == {"mp-shared", "mp-2", "mp-3"}
    assert assignment["mp-shared"] in {0, 1, 2}
    assert sum(row_counts) == len(keys)
    assert set(assignment.values()) == {0, 1, 2}


def test_negative_curated_targets_are_excluded(tmp_path: Path) -> None:
    positive = "('Cu', 'mp-1', 0)"
    negative = "('Cu', 'mp-2', 0)"
    feature_path = tmp_path / "features.npz"
    spectral_path = tmp_path / "spectra.npz"
    np.savez(
        feature_path,
        **{positive: np.zeros(64), negative: np.zeros(64)},
    )
    np.savez(
        spectral_path,
        **{
            positive: np.linspace(0.0, 1.0, 200),
            negative: np.linspace(-1e-12, 1.0, 200),
        },
    )

    keys, names = load_aligned_key_archives(feature_path, spectral_path)
    retained, targets, excluded = load_curated_targets(keys, names, spectral_path)

    assert retained == [CuratedKey("Cu", "mp-1", 0)]
    assert targets.shape == (1, 141)
    assert len(excluded) == 1
    assert excluded[0]["key"] == negative
    assert excluded[0]["minimum"] == pytest.approx(-1e-12)


def test_key_file_export_excludes_negative_targets(tmp_path: Path) -> None:
    spectral_path = tmp_path / "spectra.npz"
    key_path = tmp_path / "keys.csv.gz"
    np.savez(
        spectral_path,
        **{
            "('Cu', 'mp-1', 0)": np.zeros(200),
            "('Fe', 'mp-2', 0)": np.r_[[-1e-9], np.ones(199)],
        },
    )
    summary = export_clean_key_file(spectral_path, key_path)
    assert summary == {"source_count": 2, "key_count": 1, "excluded_negative_count": 1}
    assert load_key_file(key_path) == [CuratedKey("Cu", "mp-1", 0)]


def test_portable_target_package_round_trip(tmp_path: Path) -> None:
    key = "('Cu', 'mp-1', 0)"
    spectral_path = tmp_path / "spectra.npz"
    feature_path = tmp_path / "features.npz"
    package_path = tmp_path / "targets.npz"
    np.savez(spectral_path, **{key: np.linspace(0.0, 1.0, 200)})
    np.savez(feature_path, **{key: np.zeros(64)})

    report = create_target_package(
        spectral_path, package_path, feature_path, target_points=141
    )
    keys, targets, split_codes = load_target_package(package_path)
    assert report["row_count"] == 1
    assert report["target_provenance"] == "resampled 141"
    assert keys == [CuratedKey("Cu", "mp-1", 0)]
    assert targets.shape == (1, 141)
    assert split_codes.tolist() == [0]
    assert (tmp_path / "targets.json").is_file()

    native_path = tmp_path / "targets_200.npz"
    native_report = create_target_package(spectral_path, native_path, target_points=200)
    assert native_report["target_provenance"] == "native clean 200"
    _, native_targets, _ = load_target_package(native_path)
    assert native_targets.shape == (1, 200)


def test_key_archives_must_match(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.npz"
    spectral_path = tmp_path / "spectra.npz"
    np.savez(feature_path, **{"('Cu', 'mp-1', 0)": np.zeros(64)})
    np.savez(spectral_path, **{"('Cu', 'mp-2', 0)": np.zeros(200)})

    with pytest.raises(ExtractionError, match="key sets differ"):
        load_aligned_key_archives(feature_path, spectral_path)


def test_extract_dataset_writes_hierarchy_and_reports_ambiguity(
    tmp_path: Path,
) -> None:
    archive_path, feature_path, spectral_path, _ = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"

    report = extract_dataset(
        archive_path,
        feature_path,
        spectral_path,
        output_dir,
        expected_archive_md5=None,
    )

    assert report["complete"] is True
    assert report["row_count"] == 3
    assert report["matched_key_count"] == 3
    assert report["ambiguous_key_count"] == 1
    assert report["material_count"] == 3

    cu_site = output_dir / "extracted" / "FEFF" / "Cu" / "mp-1" / "FEFF-XANES" / "000_Cu"
    # Ambiguous keys have no unverified primary candidate: every source
    # record, including the first one, lives under candidates/.
    assert not (cu_site / "POSCAR").exists()
    assert not (cu_site / "structure.json").exists()
    assert not (cu_site / "spectrum_raw.dat").exists()
    assert (cu_site / "candidates" / "000" / "POSCAR").is_file()
    assert (cu_site / "candidates" / "000" / "structure.json").is_file()
    assert (cu_site / "candidates" / "000" / "spectrum_raw.dat").is_file()
    target = np.loadtxt(cu_site / "spectrum_141.dat")
    assert target.shape == (141, 2)
    assert target[:, 0] == pytest.approx(np.linspace(0.0, 35.0, 141))
    assert (cu_site / "candidates" / "001" / "POSCAR").is_file()
    assert (output_dir / "COMPLETE").is_file()
    assert not (output_dir / "INCOMPLETE").exists()

    # The two Cu candidates have different structures, so no material-level
    # POSCAR can safely represent both.
    assert not (output_dir / "extracted" / "FEFF" / "Cu" / "mp-1" / "POSCAR").exists()

    with np.load(output_dir / "targets_141.npz", allow_pickle=False) as targets:
        assert targets["spectras"].shape == (3, 141)
        assert targets["energies"] == pytest.approx(np.linspace(0.0, 35.0, 141))
        assert set(targets["split_codes"].tolist()) == {0, 1, 2}

    manifest = (output_dir / "manifest.csv").read_text(encoding="utf-8")
    assert "ambiguous" in manifest
    assert "missing" not in manifest
