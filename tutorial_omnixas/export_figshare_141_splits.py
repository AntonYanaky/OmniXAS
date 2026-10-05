#!/usr/bin/env python3
"""Sample Figshare raw FEFF spectra onto OmniXAS's 141-point grid and reuse its splits.

Run from the repository root with:

    python3 tutorial_omnixas/export_figshare_141_splits.py

The script reads the extracted Figshare tree in ../OmniXAS_data/figshare and
creates a deterministic fresh material-level split. It targets 80% train, 10%
validation, and 10% test site rows per element. Every valid material is assigned
once, including materials that had conflicting memberships in the old splits.
Results go to ../OmniXAS_data/figshare_141_splits.

Sampling matches the legacy window and grid: keep positive stored mu values,
use the element's configured start through start + 35 eV, then linearly sample
141 points at 0.25 eV spacing. Values outside the retained interval use the
nearest endpoint, as with numpy.interp. Non-monotonic source rows are sorted by
energy; duplicate energies are averaged.

The Figshare JSON spectra do not contain the Materials Cloud xsedge+50
normalization scalar. The per-element medians were calibrated from 4,146
crosswalked Materials Cloud TRAIN sites, and are pinned here so this script
needs no Materials Cloud raw archive at run time. Ambiguous multi-record
Figshare sites are omitted.

Requires Python 3.10+ and NumPy only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np

ELEMENT_START_EV = {
    "Ti": 4964.504,
    "V": 5464.097,
    "Cr": 5989.168,
    "Mn": 6537.886,
    "Fe": 7111.23,
    "Co": 7709.282,
    "Ni": 8332.181,
    "Cu": 8983.173,
}
WINDOW_EV = 35.0
STEP_EV = 0.25
POINT_COUNT = 141
PARTITIONS = ("train", "val", "test")
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
TARGET_SCALE_CALIBRATION = {
    "Ti": {"median": 1.662669930, "train_reference_site_count": 658, "coefficient_of_variation": 0.00817, "p05": 1.643154, "p95": 1.693106},
    "V": {"median": 1.505275248, "train_reference_site_count": 473, "coefficient_of_variation": 0.01506, "p05": 1.487556, "p95": 1.553092},
    "Cr": {"median": 1.356058304, "train_reference_site_count": 148, "coefficient_of_variation": 0.01805, "p05": 1.334948, "p95": 1.421755},
    "Mn": {"median": 1.232499460, "train_reference_site_count": 752, "coefficient_of_variation": 0.00962, "p05": 1.214039, "p95": 1.250057},
    "Fe": {"median": 1.124099787, "train_reference_site_count": 574, "coefficient_of_variation": 0.01082, "p05": 1.096120, "p95": 1.139896},
    "Co": {"median": 1.026859692, "train_reference_site_count": 352, "coefficient_of_variation": 0.01053, "p05": 1.005401, "p95": 1.040351},
    "Ni": {"median": 0.948903347, "train_reference_site_count": 451, "coefficient_of_variation": 0.00763, "p05": 0.935673, "p95": 0.959456},
    "Cu": {"median": 0.878374816, "train_reference_site_count": 738, "coefficient_of_variation": 0.00913, "p05": 0.865824, "p95": 0.891093},
}


class ExportError(RuntimeError):
    """Raised when split inputs or source spectra are inconsistent."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "OmniXAS_data"


def read_material_splits(
    repo: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    """Read per-element assignments and all global partitions per material."""
    split_dir = repo / "tutorial_omnixas" / "material_id_and_site"
    result: dict[str, dict[str, str]] = {}
    global_memberships: dict[str, set[str]] = {}
    for element in ELEMENT_START_EV:
        material_split: dict[str, str] = {}
        for partition in PARTITIONS:
            path = split_dir / f"{element}_FEFF_{partition}.txt"
            if not path.is_file():
                raise ExportError(f"Missing existing split file: {path}")
            for line_number, raw_id in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
                value = raw_id.strip()
                if not value:
                    continue
                try:
                    material_id, site = value.rsplit("_", 1)
                    int(site)
                except ValueError as exc:
                    raise ExportError(f"Invalid ID at {path}:{line_number}: {value!r}") from exc
                previous = material_split.get(material_id)
                if previous is not None and previous != partition:
                    raise ExportError(
                        f"Material leakage in existing {element} splits: {material_id} "
                        f"appears in both {previous} and {partition}"
                    )
                material_split[material_id] = partition
                global_memberships.setdefault(material_id, set()).add(partition)
        result[element] = material_split
    return result, global_memberships


def assign_material_splits(
    rows_by_material: dict[str, list[tuple[str, int, np.ndarray]]],
    seed: int = SPLIT_SEED,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Assign every material once while matching per-element site-row targets."""
    totals = {
        element: sum(
            1 for rows in rows_by_material.values() for row_element, _, _ in rows
            if row_element == element
        )
        for element in ELEMENT_START_EV
    }
    targets = {
        element: {
            partition: totals[element] * SPLIT_FRACTIONS[partition]
            for partition in PARTITIONS
        }
        for element in ELEMENT_START_EV
    }
    counts = {
        element: {partition: 0 for partition in PARTITIONS}
        for element in ELEMENT_START_EV
    }
    rng = np.random.default_rng(seed)
    materials = list(rows_by_material)
    rng.shuffle(materials)
    materials.sort(key=lambda material: -len(rows_by_material[material]))
    assignments: dict[str, str] = {}

    for material in materials:
        deltas = {
            element: sum(1 for row_element, _, _ in rows_by_material[material] if row_element == element)
            for element in ELEMENT_START_EV
        }
        scores = []
        for partition in PARTITIONS:
            score = 0.0
            for element in ELEMENT_START_EV:
                for candidate_partition in PARTITIONS:
                    candidate = counts[element][candidate_partition]
                    if candidate_partition == partition:
                        candidate += deltas[element]
                    target = targets[element][candidate_partition]
                    if target:
                        score += ((candidate - target) / target) ** 2
            scores.append(score)
        best = min(scores)
        choices = [partition for partition, score in zip(PARTITIONS, scores) if np.isclose(score, best)]
        partition = choices[int(rng.integers(len(choices)))]
        assignments[material] = partition
        for element, delta in deltas.items():
            counts[element][partition] += delta

    return assignments, counts


def sample_spectrum(path: Path, element: str, scale: float) -> np.ndarray:
    """Load source column 4 and interpolate it onto this element's 141-point grid."""
    try:
        table = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ExportError(f"Could not load raw spectrum {path}: {exc}") from exc
    if table.ndim != 2 or table.shape[1] < 4 or table.shape[0] == 0:
        raise ExportError(f"Expected a non-empty spectrum with at least 4 columns: {path}")
    energy = table[:, 0]
    mu = table[:, 3] * scale  # FEFF xmu.dat column 4; 0-based index 3.
    finite = np.isfinite(energy) & np.isfinite(mu)
    positive = finite & (mu > 0)
    energy, mu = energy[positive], mu[positive]
    if energy.size == 0:
        raise ExportError(f"No finite positive mu samples in {path}")

    order = np.argsort(energy, kind="stable")
    energy, mu = energy[order], mu[order]
    # np.interp expects increasing x. Average repeated-energy rows after sorting.
    unique_energy, starts, counts = np.unique(energy, return_index=True, return_counts=True)
    if np.any(counts > 1):
        mu = np.add.reduceat(mu, starts) / counts
        energy = unique_energy

    start = ELEMENT_START_EV[element]
    end = start + WINDOW_EV
    inside = (energy > start) & (energy < end)
    grid = np.linspace(start, end, POINT_COUNT, dtype=np.float64)
    if np.any(inside):
        energy, mu = energy[inside], mu[inside]
    else:
        # If no values fall inside the requested interval, copy the measured
        # value closest to the interval start across the target grid.
        nearest = int(np.argmin(np.abs(energy - start)))
        return np.full(POINT_COUNT, float(mu[nearest]), dtype=np.float64)
    sampled = np.interp(grid, energy, mu)
    if sampled.shape != (POINT_COUNT,) or not np.isfinite(sampled).all():
        raise ExportError(f"Sampling did not produce 141 finite values for {path}")
    return sampled


def write_rows(path: Path, rows: list[np.ndarray]) -> None:
    matrix = np.stack(rows) if rows else np.empty((0, POINT_COUNT), dtype=np.float64)
    np.savetxt(path, matrix, fmt="%.18e")


def main() -> int:
    repo = repository_root()
    root = data_root()
    source_root = root / "figshare"
    output_root = root / "figshare_141_splits"
    manifest_path = source_root / "manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing Figshare extraction manifest: {manifest_path}")
    existing_output_files = list(output_root.rglob("*")) if output_root.exists() else []
    if any(path.is_file() for path in existing_output_files):
        raise SystemExit(
            f"Output already contains files: {output_root}. "
            "Move it aside before regenerating; existing outputs are never overwritten."
        )

    try:
        _, global_memberships = read_material_splits(repo)
        globally_conflicted_materials = {
            material_id: partitions
            for material_id, partitions in global_memberships.items()
            if len(partitions) != 1
        }
        element_scales = TARGET_SCALE_CALIBRATION
        output_root.mkdir(parents=True, exist_ok=True)
        ml_dir = output_root / "ml_data"
        ids_dir = output_root / "material_id_and_site"
        ml_dir.mkdir(exist_ok=True)
        ids_dir.mkdir(exist_ok=True)

        values: dict[tuple[str, str], list[np.ndarray]] = {
            (element, partition): []
            for element in ELEMENT_START_EV
            for partition in PARTITIONS
        }
        ids: dict[tuple[str, str], list[str]] = {key: [] for key in values}
        row_metadata: dict[tuple[str, str], list[dict[str, str]]] = {key: [] for key in values}
        valid_records: dict[str, list[tuple[str, int, np.ndarray]]] = {}
        omissions: list[dict[str, str]] = []
        source_counts: dict[str, int] = {}
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"element", "material_id", "site", "status", "site_directory", "raw_candidate_count"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ExportError(f"Unexpected extraction manifest columns in {manifest_path}")
            for row in reader:
                element = row["element"]
                if element not in ELEMENT_START_EV:
                    continue
                status = row["status"]
                source_counts[status] = source_counts.get(status, 0) + 1
                material_id = row["material_id"]
                site = int(row["site"])
                key = (element, material_id, site)
                if status != "matched" or int(row["raw_candidate_count"]) != 1:
                    omissions.append({"key": repr(key), "reason": f"source_status_{status}"})
                    continue
                site_directory = PurePosixPath(row["site_directory"])
                spectrum_path = source_root.joinpath(*site_directory.parts, "spectrum_raw.dat")
                if not spectrum_path.is_file():
                    omissions.append({"key": repr(key), "reason": "raw_spectrum_file_missing"})
                    continue
                try:
                    scale = element_scales[element]["median"]
                    y = sample_spectrum(spectrum_path, element, scale)
                except ExportError as exc:
                    omissions.append({"key": repr(key), "reason": str(exc)})
                    continue
                valid_records.setdefault(material_id, []).append((element, site, y))

        split_by_material, assigned_counts = assign_material_splits(valid_records)
        conflicting_material_site_count = sum(
            len(valid_records[material_id])
            for material_id in globally_conflicted_materials
            if material_id in valid_records
        )
        figshare_only_material_site_count = sum(
            len(rows)
            for material_id, rows in valid_records.items()
            if material_id not in global_memberships
        )
        for material_id, rows in valid_records.items():
            partition = split_by_material[material_id]
            for element, site, y in rows:
                split_key = (element, partition)
                ids[split_key].append(f"{material_id}_{site:03d}")
                values[split_key].append(y)
                row_metadata[split_key].append({
                    "split_source": "fresh_material_split",
                    "target_scale_factor": f"{element_scales[element]['median']:.12g}",
                })

        output_counts: dict[str, dict[str, int]] = {}
        index_rows: list[dict[str, object]] = []
        for element in ELEMENT_START_EV:
            output_counts[element] = {}
            for partition in PARTITIONS:
                key = (element, partition)
                order = sorted(range(len(ids[key])), key=lambda index: ids[key][index])
                sorted_ids = [ids[key][index] for index in order]
                sorted_values = [values[key][index] for index in order]
                sorted_metadata = [row_metadata[key][index] for index in order]
                if len(sorted_ids) != len(set(sorted_ids)):
                    raise ExportError(f"Duplicate Figshare site ID in {element} {partition}")
                write_rows(ml_dir / f"{element}_FEFF_{partition}_y.txt", sorted_values)
                (ids_dir / f"{element}_FEFF_{partition}.txt").write_text(
                    "".join(f"{value}\n" for value in sorted_ids), encoding="utf-8"
                )
                output_counts[element][partition] = len(sorted_ids)
                for index, site_id in enumerate(sorted_ids):
                    index_rows.append({
                        "element": element,
                        "split": partition,
                        "row_index": index,
                        "material_id_and_site": site_id,
                        **sorted_metadata[index],
                    })

        # Verify that every fresh material assignment is globally disjoint.
        figshare_materials_by_partition = {
            partition: {
                material_id
                for material_id, row_partition in split_by_material.items()
                if row_partition == partition
            }
            for partition in PARTITIONS
        }
        assigned_materials = set().union(*figshare_materials_by_partition.values())
        if assigned_materials != set(valid_records):
            raise ExportError("Fresh split assignment did not cover every valid material")
        for index, first in enumerate(PARTITIONS):
            for second in PARTITIONS[index + 1 :]:
                overlap = figshare_materials_by_partition[first] & figshare_materials_by_partition[second]
                if overlap:
                    raise ExportError(
                        f"Fresh material split has an overlap between {first} and {second}: "
                        f"{sorted(overlap)[:5]}"
                    )

        with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "element", "split", "row_index", "material_id_and_site",
                    "split_source", "target_scale_factor",
                ),
            )
            writer.writeheader()
            writer.writerows(index_rows)
        with (output_root / "omitted_sites.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("key", "reason"))
            writer.writeheader()
            writer.writerows(omissions)

        report = {
            "source": str(source_root.resolve()),
            "existing_split_ids": str((repo / "tutorial_omnixas" / "material_id_and_site").resolve()),
            "output": str(output_root.resolve()),
            "elements": list(ELEMENT_START_EV),
            "target_grid": {
                "points": POINT_COUNT,
                "step_ev": STEP_EV,
                "window_ev": WINDOW_EV,
                "element_start_ev": ELEMENT_START_EV,
                "endpoint_policy": "numpy.interp endpoint carry-forward; constant nearest measured value if no source points fall inside the window",
            },
            "split_policy": "Create a fresh deterministic material-level split with seed 42. Assign every valid material exactly once. Greedily minimize normalized per-element site-row error against 80% train, 10% validation, and 10% test targets. Old split memberships, including conflicts, do not control assignment.",
            "target_policy": "Use positive Figshare raw spectrum column 4 and multiply by a per-element median scale estimated from matched Materials Cloud train records only (xsedge+50 / Bohr-radius-squared * 1000), then interpolate to 141 points",
            "source_manifest_status_counts": source_counts,
            "materialscloud_train_scale_calibration": element_scales,
            "split_seed": SPLIT_SEED,
            "target_fractions": SPLIT_FRACTIONS,
            "assigned_site_rows_by_element_and_split": assigned_counts,
            "figshare_only_material_site_rows": figshare_only_material_site_count,
            "previously_conflicting_material_site_rows_assigned": conflicting_material_site_count,
            "previously_conflicting_material_count": len(globally_conflicted_materials),
            "written_rows_by_element_and_split": output_counts,
            "omitted_site_count": len(omissions),
        }
        (output_root / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, ExportError) as exc:
        print(f"Figshare split export failed: {exc}", file=sys.stderr)
        return 2

    print(f"Figshare 141-point split export complete: {output_root.resolve()}")
    for element, counts in output_counts.items():
        print(f"{element}: " + ", ".join(f"{partition}={counts[partition]}" for partition in PARTITIONS))
    print(f"Omitted site rows: {len(omissions)} (see omitted_sites.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
