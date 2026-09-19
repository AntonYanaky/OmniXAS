#!/usr/bin/env python3
"""Build the versioned eight-element FEFF target package (v2) from raw Figshare records.

This curator rebuilds the OmniXAS eight-element K-edge dataset directly from the
raw Figshare ``xas.json.tgz`` archive. It does not read the clean or final NPZ
files, so no provenance from those files is inherited.

Curation policy (applied in this order):

1. Keep K-edge XANES records for Ti, V, Cr, Mn, Fe, Co, Ni, Cu.
2. Identity is ``(element, material_id, site)`` from cross-checked metadata.
3. Duplicate records for one identity are accepted only when all processed
   curves agree within ``--duplicate-tol``; otherwise the key is rejected.
4. One material must map to exactly one structure digest; materials whose
   retained records show more than one structure are rejected.
5. Symmetry-equivalent absorber sites are collapsed to one representative
   (lowest site index) when their processed curves agree within
   ``--symmetry-tol``; disagreeing groups are rejected.
6. Pointwise 2.5-sigma outlier filtering per element (paper FEFF rule).
7. Spectra with any target value below ``--min-target`` are rejected.
8. Material-level 80/10/10 splits (deterministic, seeded).

Target definition (calibrated against the community clean package):

    target[i] = interp(raw[:, 0], raw[:, 3],
                       E_start(element) + edge_offset(element) + grid[i])

with ``grid = linspace(0, 35, 141)`` (0.25 eV spacing). ``E_start`` values are
Table S1 of the OmniXAS paper (arXiv:2409.19552). The calibrated edge offsets
reproduce the established curated targets exactly (relative residual 0.000)
for Ti, V, Fe, Co, and Ni. Cr, Mn, and Cu have no trustworthy curated
reference (known identity-indexing defects), so they use the same default
offset. Rows are sorted by energy before interpolation; records with
duplicate energy rows are rejected.

Output layout (the package acts as a showcase-style data root):

    <output>/tutorial_omnixas/ml_data/<task>_<split>_y.txt
    <output>/tutorial_omnixas/material_id_and_site/<task>_<split>.txt
    <output>/extracted/FEFF/<element>/<material_id>/POSCAR
    <output>/targets_141.npz
    <output>/manifest.csv
    <output>/rejections.csv
    <output>/symmetry_groups.csv
    <output>/report.json
    <output>/README.md
    <output>/PACKAGE_COMPLETE.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from extract_figshare_anionxas import (
    CuratedKey,
    ExtractionError,
    assign_material_splits,
    canonical_structure_digest,
    file_digest,
    iter_json_lines,
    record_material_site,
    record_spectrum_table,
    record_structure,
    site_element,
    structure_to_poscar,
)

FIGSHARE_ARTICLE_ID = 5678998
ELEMENTS = ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu")
SPLIT_NAMES = ("train", "val", "test")

# Table S1 of arXiv:2409.19552 (starting energy points of XANES spectra).
TABLE_S1_STARTS = {
    "Ti": 4964.504,
    "V": 5464.097,
    "Cr": 5989.168,
    "Mn": 6537.886,
    "Fe": 7111.230,
    "Co": 7709.282,
    "Ni": 8332.181,
    "Cu": 8983.173,
}
DEFAULT_EDGE_OFFSET = 1.35
CALIBRATED_EDGE_OFFSETS = {"Ni": 1.40}
MU_COLUMN = 3
ENERGY_SPAN_EV = 35.0
TARGET_POINTS = 141
PACKAGE_FORMAT_VERSION = 2
PACKAGE_COMPLETE = "PACKAGE_COMPLETE.json"
OUTLIER_SIGMA = 2.5
MIN_TARGET = -1e-3
DUPLICATE_TOL = 1e-3
SYMMETRY_TOL = 0.05
SYMMETRY_SYMPREC = 1e-2


class CurationError(RuntimeError):
    """Raised when curation cannot continue."""


@dataclass
class Candidate:
    """One raw record for an identity key."""

    line: int
    digest: str
    spectrum: np.ndarray


@dataclass
class CurationResult:
    """Retained rows plus curation artifacts."""

    keys: list[CuratedKey]
    spectra: dict[CuratedKey, np.ndarray]
    source_lines: dict[CuratedKey, int]
    split_by_material: dict[str, int]
    material_digests: dict[str, str]
    material_digest_counts: dict[str, int]
    symmetry_groups: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def default_edge_offsets() -> dict[str, float]:
    """Return the calibrated per-element edge offsets."""
    return {element: CALIBRATED_EDGE_OFFSETS.get(element, DEFAULT_EDGE_OFFSET) for element in ELEMENTS}


def curve_agrees(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    """Compare two processed curves with a max-abs-difference tolerance."""
    return bool(np.max(np.abs(first - second)) <= tolerance)


def spectrum_to_target(
    table: np.ndarray,
    element: str,
    edge_offsets: Mapping[str, float],
    energy_span: float = ENERGY_SPAN_EV,
    target_points: int = TARGET_POINTS,
) -> tuple[np.ndarray, float]:
    """Resample one raw spectrum table onto the absolute v2 target grid."""
    energy = table[:, 0]
    intensity = table[:, MU_COLUMN]
    order = np.argsort(energy, kind="stable")
    energy = energy[order]
    intensity = intensity[order]
    if len(np.unique(energy)) != len(energy):
        raise ExtractionError("Raw spectrum has duplicate energy rows")
    e_start = TABLE_S1_STARTS[element] + edge_offsets[element]
    grid = e_start + np.linspace(0.0, energy_span, target_points)
    return np.interp(grid, energy, intensity).astype(np.float32), e_start


def parse_elements(value: str | None) -> tuple[str, ...]:
    """Parse and validate the requested element subset."""
    if not value:
        return ELEMENTS
    elements = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [element for element in elements if element not in TABLE_S1_STARTS]
    if unknown:
        raise CurationError(f"Unknown elements: {unknown}; supported: {list(TABLE_S1_STARTS)}")
    return elements


def scan_pass(
    archive: Path,
    elements: tuple[str, ...],
    edge_offsets: Mapping[str, float],
    limit: int | None,
) -> tuple[dict[CuratedKey, list[Candidate]], dict[str, dict[str, int]], dict[str, int]]:
    """Stream the archive and collect processed candidates for the requested elements."""
    candidates: dict[CuratedKey, list[Candidate]] = {}
    material_digests: dict[str, dict[str, int]] = {}
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    seen_records = 0
    for line_number, record in iter_json_lines(archive):
        seen_records += 1
        if limit is not None and seen_records > limit:
            break
        try:
            material_id, site = record_material_site(record)
            structure = record_structure(record)
            element = site_element(structure, site)
            if element not in elements:
                continue
            table = record_spectrum_table(record)
            spectrum, _ = spectrum_to_target(table, element, edge_offsets)
            digest = canonical_structure_digest(structure)
        except ExtractionError as exc:
            reject(f"record_parse_error: {exc}")
            continue

        key = CuratedKey(element=element, material_id=material_id, site=site)
        candidates.setdefault(key, []).append(Candidate(line=line_number, digest=digest, spectrum=spectrum))
        material_digests.setdefault(material_id, {})
        material_digests[material_id][digest] = material_digests[material_id].get(digest, 0) + 1
    return candidates, material_digests, rejected


def resolve_duplicate_candidates(
    candidates: dict[CuratedKey, list[Candidate]],
    duplicate_tol: float,
) -> tuple[dict[CuratedKey, Candidate], list[dict]]:
    """Keep one candidate per key only when every processed curve and structure agrees."""
    retained: dict[CuratedKey, Candidate] = {}
    rejections: list[dict] = []
    for key in sorted(candidates):
        group = candidates[key]
        reference = group[0].spectrum
        reference_digest = group[0].digest
        curve_disagreements = sum(
            1 for candidate in group[1:] if not curve_agrees(reference, candidate.spectrum, duplicate_tol)
        )
        if curve_disagreements:
            rejections.append(
                {
                    "element": key.element,
                    "material_id": key.material_id,
                    "site": key.site,
                    "reason": "duplicate_curve_conflict",
                    "detail": f"{curve_disagreements}/{len(group) - 1} duplicate curves disagree beyond {duplicate_tol}",
                }
            )
            continue
        structure_disagreements = sum(
            1 for candidate in group[1:] if candidate.digest != reference_digest
        )
        if structure_disagreements:
            rejections.append(
                {
                    "element": key.element,
                    "material_id": key.material_id,
                    "site": key.site,
                    "reason": "duplicate_structure_conflict",
                    "detail": f"{structure_disagreements}/{len(group) - 1} duplicate records disagree on the structure digest",
                }
            )
            continue
        retained[key] = min(group, key=lambda candidate: candidate.line)
    return retained, rejections


def reject_structure_conflicts(
    retained: dict[CuratedKey, Candidate],
) -> list[dict]:
    """Reject every row of a material whose retained records disagree on the structure."""
    digests_by_material: dict[str, set[str]] = {}
    for key, candidate in retained.items():
        digests_by_material.setdefault(key.material_id, set()).add(candidate.digest)
    rejected_materials = {
        material_id for material_id, digests in digests_by_material.items() if len(digests) > 1
    }
    rejections = [
        {
            "element": "",
            "material_id": material_id,
            "site": -1,
            "reason": "structure_conflict",
            "detail": f"{len(digests_by_material[material_id])} distinct structure digests",
        }
        for material_id in sorted(rejected_materials)
    ]
    for key in list(retained):
        if key.material_id in rejected_materials:
            del retained[key]
    return rejections


def analyze_equivalence_groups(structure: Mapping[str, object]) -> list[list[int]]:
    """Return symmetry-equivalent site-index groups for one material structure."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    symmetrized = SpacegroupAnalyzer(Structure.from_dict(dict(structure)), symprec=SYMMETRY_SYMPREC)
    return sorted(
        (sorted(int(index) for index in group) for group in symmetrized.get_symmetrized_structure().equivalent_indices),
        key=lambda members: members[0],
    )


def structure_pass(
    archive: Path,
    retained: dict[CuratedKey, Candidate],
    output_dir: Path,
    use_symmetry: bool,
    strict_symmetry: bool,
    limit: int | None,
) -> tuple[dict[str, list[list[int]]], list[dict], dict[str, int]]:
    """Stream the archive again: write POSCARs and analyze symmetry per material.

    Returns the equivalence groups per material, rejections, and warning counts.
    """
    first_line_by_material: dict[str, int] = {}
    for key, candidate in retained.items():
        current = first_line_by_material.get(key.material_id)
        if current is None or candidate.line < current:
            first_line_by_material[key.material_id] = candidate.line
    pending = set(first_line_by_material.values())

    digest_by_line = {candidate.line: candidate.digest for candidate in retained.values()}
    groups_by_material: dict[str, list[list[int]]] = {}
    rejections: list[dict] = []
    warnings: dict[str, int] = {}
    material_of_line: dict[int, str] = {}
    for key, candidate in retained.items():
        material_of_line[candidate.line] = key.material_id

    def warn(reason: str) -> None:
        warnings[reason] = warnings.get(reason, 0) + 1

    seen = 0
    for line_number, record in iter_json_lines(archive):
        seen += 1
        if limit is not None and seen > limit:
            break
        if line_number not in pending:
            continue
        material_id = material_of_line[line_number]
        try:
            structure = record_structure(record)
            digest = canonical_structure_digest(structure)
            if digest != digest_by_line[line_number]:
                raise ExtractionError("Structure digest changed between passes")
            source_site = next(
                key.site for key, candidate in retained.items() if candidate.line == line_number
            )
            poscar_text = structure_to_poscar(structure, material_id, source_site)
        except ExtractionError as exc:
            rejections.append(
                {"element": "", "material_id": material_id, "site": -1, "reason": "structure_pass_error", "detail": str(exc)}
            )
            for key in list(retained):
                if key.material_id == material_id:
                    del retained[key]
            continue
        for element in {key.element for key in retained if key.material_id == material_id}:
            material_dir = output_dir / "extracted" / "FEFF" / element / material_id
            material_dir.mkdir(parents=True, exist_ok=True)
            (material_dir / "POSCAR").write_text(poscar_text, encoding="utf-8")
        if use_symmetry:
            try:
                groups_by_material[material_id] = analyze_equivalence_groups(structure)
            except Exception as exc:  # noqa: BLE001 - degrade to singletons or reject
                if strict_symmetry:
                    rejections.append(
                        {
                            "element": "",
                            "material_id": material_id,
                            "site": -1,
                            "reason": "symmetry_analysis_failed",
                            "detail": str(exc),
                        }
                    )
                    for key in list(retained):
                        if key.material_id == material_id:
                            del retained[key]
                    continue
                warn("symmetry_degraded_to_singletons")
                groups_by_material[material_id] = []
        pending.discard(line_number)
        if not pending:
            break

    missed = sorted(
        {key.material_id for key in retained}
        - {material_of_line[candidate.line] for candidate in retained.values()}
        - {row["material_id"] for row in rejections}
    )
    for material_id in missed:
        rejections.append(
            {
                "element": "",
                "material_id": material_id,
                "site": -1,
                "reason": "structure_not_found_in_second_pass",
                "detail": "no record matched in the structure pass",
            }
        )
        for key in list(retained):
            if key.material_id == material_id:
                del retained[key]
    return groups_by_material, rejections, warnings


def collapse_symmetry(
    retained: dict[CuratedKey, Candidate],
    groups_by_material: Mapping[str, list[list[int]]],
    symmetry_tol: float,
) -> tuple[dict[CuratedKey, Candidate], list[dict], list[dict]]:
    """Collapse symmetry-equivalent absorber sites to one verified representative."""
    by_material: dict[tuple[str, str], list[CuratedKey]] = {}
    for key in retained:
        by_material.setdefault((key.element, key.material_id), []).append(key)

    collapsed: dict[CuratedKey, Candidate] = {}
    group_rows: list[dict] = []
    rejections: list[dict] = []
    for (element, material_id), keys in sorted(by_material.items()):
        equivalence = groups_by_material.get(material_id, [])
        requested = {key.site for key in keys}
        groups: list[list[int]] = []
        for members in equivalence:
            overlap = sorted(requested & set(members))
            if overlap:
                groups.append(overlap)
        grouped_sites = {site for members in groups for site in members}
        groups.extend([site] for site in sorted(requested - grouped_sites))

        for members in groups:
            group_keys = [key for key in keys if key.site in members]
            reference = retained[min(group_keys, key=lambda key: key.site)].spectrum
            max_diff = max(
                float(np.max(np.abs(retained[key].spectrum - reference))) for key in group_keys
            )
            if len(members) > 1 and max_diff > symmetry_tol:
                rejections.append(
                    {
                        "element": element,
                        "material_id": material_id,
                        "site": -1,
                        "reason": "symmetry_group_conflict",
                        "detail": f"sites {members} disagree, max diff {max_diff:.4g} > {symmetry_tol}",
                    }
                )
                continue
            representative = min(group_keys, key=lambda key: key.site)
            collapsed[representative] = retained[representative]
            group_rows.append(
                {
                    "element": element,
                    "material_id": material_id,
                    "representative_site": representative.site,
                    "group_sites_list": members,
                    "group_size": len(members),
                    "max_abs_diff": max_diff,
                }
            )
    return collapsed, group_rows, rejections


def pointwise_outlier_filter(
    retained: dict[CuratedKey, Candidate],
    sigma: float,
    min_target: float,
) -> tuple[dict[CuratedKey, Candidate], dict[str, dict], list[dict]]:
    """Apply the paper's pointwise sigma rule and the negative-target rule."""
    by_element: dict[str, list[CuratedKey]] = {}
    for key in retained:
        by_element.setdefault(key.element, []).append(key)

    stats: dict[str, dict] = {}
    rejections: list[dict] = []
    rejected_keys: set[CuratedKey] = set()
    for element in sorted(by_element):
        keys = by_element[element]
        stack = np.stack([retained[key].spectrum for key in keys])
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        outlier_count = 0
        negative_count = 0
        for key in keys:
            spectrum = retained[key].spectrum
            safe_std = std > 0
            if np.any(safe_std & (np.abs(spectrum - mean) > sigma * std)):
                rejections.append(
                    {
                        "element": element,
                        "material_id": key.material_id,
                        "site": key.site,
                        "reason": "pointwise_outlier",
                        "detail": f"beyond {sigma} sigma at some grid point",
                    }
                )
                rejected_keys.add(key)
                outlier_count += 1
                continue
            minimum = float(spectrum.min())
            if minimum < min_target:
                rejections.append(
                    {
                        "element": element,
                        "material_id": key.material_id,
                        "site": key.site,
                        "reason": "negative_target",
                        "detail": f"min {minimum:.4g} < {min_target}",
                    }
                )
                rejected_keys.add(key)
                negative_count += 1
        stats[element] = {
            "before_filter": len(keys),
            "removed_pointwise_outlier": outlier_count,
            "removed_negative_target": negative_count,
            "kept": len(keys) - outlier_count - negative_count,
        }
    return {key: retained[key] for key in retained if key not in rejected_keys}, stats, rejections


def write_split_files(
    output_dir: Path,
    keys: list[CuratedKey],
    spectra: dict[CuratedKey, np.ndarray],
    split_by_material: Mapping[str, int],
) -> None:
    """Write the showcase-style y and ID files for every task/split."""
    data_dir = output_dir / "tutorial_omnixas" / "ml_data"
    id_dir = output_dir / "tutorial_omnixas" / "material_id_and_site"
    data_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)
    for element in ELEMENTS:
        task = f"{element}_FEFF"
        element_keys = sorted(key for key in keys if key.element == element)
        for split_code, split in enumerate(("train", "val", "test")):
            split_keys = [key for key in element_keys if split_by_material[key.material_id] == split_code]
            y_path = data_dir / f"{task}_{split}_y.txt"
            id_path = id_dir / f"{task}_{split}.txt"
            if not split_keys:
                np.savetxt(y_path, np.zeros((0, TARGET_POINTS)))
                id_path.write_text("", encoding="utf-8")
                continue
            np.savetxt(y_path, np.stack([spectra[key] for key in split_keys]), fmt="%.18e")
            id_path.write_text(
                "".join(f"{key.material_id}_{key.site}\n" for key in split_keys), encoding="utf-8"
            )


def write_package(
    output_dir: Path,
    result: CurationResult,
    args: argparse.Namespace,
    edge_offsets: Mapping[str, float],
    archive_digest: str,
    material_digest_counts: Mapping[str, int],
) -> None:
    """Write the versioned v2 target package."""
    if not result.keys:
        raise CurationError("No rows survived curation; refusing to write an empty package")
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_matrix = np.stack([result.spectra[key] for key in result.keys])
    split_codes = np.array([result.split_by_material[key.material_id] for key in result.keys], dtype=np.int64)
    np.savez_compressed(
        output_dir / "targets_141.npz",
        elements=np.array([key.element for key in result.keys]),
        material_ids=np.array([key.material_id for key in result.keys]),
        sites=np.array([key.site for key in result.keys], dtype=np.int64),
        energies=np.linspace(0.0, ENERGY_SPAN_EV, TARGET_POINTS),
        spectras=spectra_matrix,
        split_codes=split_codes,
    )

    write_split_files(output_dir, result.keys, result.spectra, result.split_by_material)

    group_by_key: dict[CuratedKey, dict] = {}
    for row in result.symmetry_groups:
        for site in row["group_sites_list"]:
            group_by_key[CuratedKey(element=row["element"], material_id=row["material_id"], site=site)] = row

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "element",
                "material_id",
                "site",
                "split",
                "source_line",
                "structure_digest",
                "symmetry_group_size",
            ],
        )
        writer.writeheader()
        for key in result.keys:
            row = group_by_key.get(key)
            writer.writerow(
                {
                    "element": key.element,
                    "material_id": key.material_id,
                    "site": key.site,
                    "split": ("train", "val", "test")[result.split_by_material[key.material_id]],
                    "source_line": result.source_lines[key],
                    "structure_digest": result.material_digests[key.material_id][:16],
                    "symmetry_group_size": row["group_size"] if row else 1,
                }
            )

    with (output_dir / "rejections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["element", "material_id", "site", "reason", "detail"])
        writer.writeheader()
        writer.writerows(result.rejections)

    with (output_dir / "symmetry_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["element", "material_id", "representative_site", "group_sites", "group_size", "max_abs_diff"],
        )
        writer.writeheader()
        for row in result.symmetry_groups:
            writer.writerow(
                {
                    "element": row["element"],
                    "material_id": row["material_id"],
                    "representative_site": row["representative_site"],
                    "group_sites": " ".join(str(site) for site in row["group_sites_list"]),
                    "group_size": row["group_size"],
                    "max_abs_diff": row["max_abs_diff"],
                }
            )

    split_counts = {
        split: int(np.count_nonzero(split_codes == code)) for code, split in enumerate(("train", "val", "test"))
    }
    report = {
        "format_version": PACKAGE_FORMAT_VERSION,
        "figshare_article_id": FIGSHARE_ARTICLE_ID,
        "archive_digest_sha256": archive_digest,
        "target_definition": {
            "intensity_column": MU_COLUMN,
            "energy_span_ev": ENERGY_SPAN_EV,
            "target_points": TARGET_POINTS,
            "grid_spacing_ev": ENERGY_SPAN_EV / (TARGET_POINTS - 1),
            "table_s1_starts": TABLE_S1_STARTS,
            "edge_offsets": dict(edge_offsets),
            "rule": (
                "target = interp(raw energy, raw mu, E_start + edge_offset + linspace(0, 35, 141)); "
                "rows sorted by energy; duplicate energy rows rejected"
            ),
            "calibration": (
                "Reproduces the community clean targets exactly (relative residual 0.000) for Ti, V, Fe, "
                "Co, Ni; Cr, Mn, Cu have no trustworthy curated reference."
            ),
        },
        "policy": {
            "duplicate_tolerance": args.duplicate_tol,
            "symmetry_tolerance": args.symmetry_tol,
            "outlier_sigma": args.sigma,
            "min_target": args.min_target,
            "symmetry_enabled": not args.no_symmetry,
            "strict_symmetry": args.strict_symmetry,
            "seed": args.seed,
        },
        "element_stats": result.stats,
        "split_row_counts": split_counts,
        "materials_with_multiple_structure_digests": {
            material_id: count for material_id, count in material_digest_counts.items() if count > 1
        },
        "counts": {
            "retained_rows": len(result.keys),
            "rejections": len(result.rejections),
            "materials": len(result.material_digests),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / PACKAGE_COMPLETE).write_text(
        json.dumps({"status": "complete", "format_version": PACKAGE_FORMAT_VERSION}, indent=2),
        encoding="utf-8",
    )


def build_readme(output_dir: Path) -> None:
    """Write the package README."""
    readme = f"""# AnionXAS FEFF v2 target package

Rebuilt from raw Figshare article {FIGSHARE_ARTICLE_ID} records by
`tutorial_omnixas/extract_figshare_feff_v2.py`. See `report.json` for the
full curation policy, calibration, and per-element counts.

- Targets: {TARGET_POINTS} points, 0-35 eV, 0.25 eV spacing, absolute per-element
  starts (Table S1 + calibrated edge offset), intensity = raw column {MU_COLUMN}.
- Rows: one per identity key after duplicate rejection, structure-conflict
  rejection, symmetry collapse, pointwise {OUTLIER_SIGMA}-sigma filtering, and
  negative-target rejection.
- Splits: material-level 80/10/10, deterministic seed, `split_codes` 0/1/2.
- The `tutorial_omnixas/ml_data` and `tutorial_omnixas/material_id_and_site`
  subdirectories follow the showcase layout so
  `train_m3gnet_xas_pipeline_v2.py` can consume the package directly.
- Structures: one POSCAR per (element, material) under
  `extracted/FEFF/<element>/<material_id>`; site indices refer to the archived
  structure site order. `structure_path` in `omnixas/data/feff_graph.py`
  resolves this layout directly.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def curate(args: argparse.Namespace) -> CurationResult:
    """Run the full v2 curation pipeline."""
    archive = Path(args.archive)
    if not archive.is_file():
        raise CurationError(f"Archive not found: {archive}")
    elements = parse_elements(args.elements)
    edge_offsets = default_edge_offsets()
    if args.edge_offset is not None:
        edge_offsets = {element: args.edge_offset for element in ELEMENTS}

    print(f"[pass 1] scanning {archive} for elements {elements}", flush=True)
    candidates, material_digest_counts, scan_rejected = scan_pass(archive, elements, edge_offsets, args.limit)
    total_candidates = sum(len(group) for group in candidates.values())
    print(f"[pass 1] {total_candidates} candidate records for {len(candidates)} identity keys", flush=True)
    print(f"[pass 1] rejected records: {scan_rejected}", flush=True)

    retained, rejections = resolve_duplicate_candidates(candidates, args.duplicate_tol)
    print(f"[dedup] retained {len(retained)} keys, rejected {len(rejections)}", flush=True)

    rejections += reject_structure_conflicts(retained)
    print(f"[structures] retained {len(retained)} keys after structure-conflict rejection", flush=True)

    groups_by_material: dict[str, list[list[int]]] = {}
    # The structure pass always writes POSCARs; symmetry analysis is optional.
    groups_by_material, structure_rejections, warnings = structure_pass(
        archive,
        retained,
        Path(args.output_dir),
        use_symmetry=not args.no_symmetry,
        strict_symmetry=args.strict_symmetry,
        limit=args.limit,
    )
    rejections += structure_rejections
    print(f"[structures] wrote POSCARs, {len(warnings)} warnings, {len(structure_rejections)} rejections", flush=True)

    retained, group_rows, symmetry_rejections = collapse_symmetry(retained, groups_by_material, args.symmetry_tol)
    rejections += symmetry_rejections
    print(f"[symmetry] collapsed to {len(retained)} keys, rejected {len(symmetry_rejections)}", flush=True)

    retained, stats, filter_rejections = pointwise_outlier_filter(retained, args.sigma, args.min_target)
    rejections += filter_rejections
    print(f"[outliers] retained {len(retained)} keys after sigma and negative filtering", flush=True)

    keys = sorted(retained)
    split_by_material, _ = assign_material_splits(keys, args.seed)
    material_digests: dict[str, str] = {}
    for key in keys:
        material_digests.setdefault(key.material_id, retained[key].digest)
    source_lines = {key: retained[key].line for key in keys}

    return CurationResult(
        keys=keys,
        spectra={key: retained[key].spectrum for key in keys},
        source_lines=source_lines,
        split_by_material=split_by_material,
        material_digests=material_digests,
        material_digest_counts={mid: len(digests) for mid, digests in material_digest_counts.items()},
        symmetry_groups=group_rows,
        rejections=rejections,
        stats=stats,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the v2 curation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Path to xas.json.tgz")
    parser.add_argument("--output-dir", required=True, help="Package output directory")
    parser.add_argument("--elements", default=None, help="Comma-separated subset of Ti,V,Cr,Mn,Fe,Co,Ni,Cu")
    parser.add_argument("--sigma", type=float, default=OUTLIER_SIGMA, help="Pointwise outlier sigma")
    parser.add_argument("--min-target", type=float, default=MIN_TARGET, help="Reject spectra below this value")
    parser.add_argument("--duplicate-tol", type=float, default=DUPLICATE_TOL, help="Max abs diff for duplicate curves")
    parser.add_argument("--symmetry-tol", type=float, default=SYMMETRY_TOL, help="Max abs diff within a symmetry group")
    parser.add_argument("--edge-offset", type=float, default=None, help="Override all calibrated edge offsets")
    parser.add_argument("--no-symmetry", action="store_true", help="Disable symmetry collapse")
    parser.add_argument("--strict-symmetry", action="store_true", help="Reject materials whose symmetry analysis fails")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Read at most this many records (smoke tests)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CurationError(f"Output directory is not empty: {output_dir}")

    result = curate(args)
    archive_digest = file_digest(Path(args.archive), "sha256")
    write_package(output_dir, result, args, default_edge_offsets(), archive_digest, result.material_digest_counts)
    build_readme(output_dir)
    print(f"Package complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
