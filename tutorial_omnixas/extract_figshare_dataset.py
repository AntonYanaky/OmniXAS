#!/usr/bin/env python3
"""Stream Figshare records into the OmniXAS_data/figshare directory layout.

The output mirrors the desktop extraction's FEFF tree and its site-level files:

    FEFF/<element>/<material>/FEFF-XANES/<site>_<element>/

Only XANES records are written. ELNES records are counted for provenance but
excluded because the desktop Figshare tree contains XANES. Differing XANES
records for one key are preserved under ``candidates/``; exact repeats with the
same raw table, rounded geometry, and FEFF inputs are collapsed to the first source row.
No energy alignment, interpolation, rescaling, or split assignment is performed.

Requires Python 3.10+ and NumPy. Uses only the Python standard library and NumPy,
so it runs on Rocky Linux without a FEFF, pymatgen, or GPU installation.

Run on Rocky Linux with one command::

    python3 tutorial_omnixas/extract_figshare_dataset.py

The script downloads Figshare file 9932248, checks its published MD5, and extracts raw XANES records under `../OmniXAS_data/figshare` beside the repository containing this script. It needs Python 3.10+ and NumPy. The separate final-target NPZ is not part of the Figshare article, so this raw-data extraction does not create `spectrum_final_npz.npy` files.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import BinaryIO, Iterator, Mapping, Sequence
from urllib.request import urlopen

import numpy as np


ARTICLE_ID = 5678998
ARCHIVE_FILE_ID = 9932248
EXPECTED_ARCHIVE_MD5 = "e866677ebb9270aeb2e15c725bef7e05"
ARCHIVE_URL = f"https://ndownloader.figshare.com/files/{ARCHIVE_FILE_ID}"
ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
MATERIAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SITE_FILES = (
    "figshare_record.json",
    "structure.json",
    "POSCAR",
    "spectrum_raw.dat",
    "spectrum_raw_metadata.json",
)
MANIFEST_FIELDS = (
    "row_index",
    "key",
    "element",
    "material_id",
    "site",
    "split",
    "raw_candidate_count",
    "first_source_line",
    "status",
    "site_directory",
    "raw_candidate_count_before_cleanup",
)


class ExtractionError(RuntimeError):
    """Raised when the source files cannot produce a consistent extraction."""


Key = tuple[str, str, int]


def key_text(key: Key) -> str:
    return repr(key)


def safe_site_directory(root: Path, key: Key) -> Path:
    element, material, site = key
    return root / "FEFF" / element / material / "FEFF-XANES" / f"{site:03d}_{element}"


def _consistent_text(label: str, values: Sequence[object]) -> str:
    present = [str(value) for value in values if value is not None and str(value) != ""]
    if not present:
        raise ExtractionError(f"Source record has no {label}")
    if len(set(present)) != 1:
        raise ExtractionError(f"Source record has conflicting {label}: {present!r}")
    return present[0]


def structure_for(record: Mapping[str, object]) -> Mapping[str, object]:
    structure = record.get("structure")
    spectrum = record.get("spectrum")
    if structure is None and isinstance(spectrum, Mapping):
        structure = spectrum.get("structure")
    if not isinstance(structure, Mapping):
        raise ExtractionError("Source record has no structure object")
    return structure


def source_key(record: Mapping[str, object]) -> Key:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    spectrum = record.get("spectrum")
    if not isinstance(spectrum, Mapping):
        spectrum = {}
    material = _consistent_text(
        "material ID",
        (
            record.get("mp_id"),
            record.get("material_id"),
            metadata.get("mp_id"),
            metadata.get("material_id"),
            spectrum.get("material_id"),
        ),
    )
    if not MATERIAL_RE.fullmatch(material):
        raise ExtractionError(f"Unsafe material ID: {material!r}")

    site_values = [
        value
        for value in (
            record.get("absorbing_atom"),
            metadata.get("absorbing_atom_index"),
            spectrum.get("absorbing_index"),
            spectrum.get("absorbing_atom"),
        )
        if value is not None
    ]
    if not site_values:
        raise ExtractionError("Source record has no absorbing site index")
    try:
        site_indices = {int(value) for value in site_values}
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"Invalid absorbing site index: {site_values!r}") from exc
    if len(site_indices) != 1:
        raise ExtractionError(f"Conflicting absorbing site indices: {site_values!r}")
    site = site_indices.pop()
    if site < 0:
        raise ExtractionError(f"Negative absorbing site index: {site}")

    structure = structure_for(record)
    sites = structure.get("sites")
    if not isinstance(sites, list) or site >= len(sites):
        raise ExtractionError(f"Absorbing site {site} is outside the source structure")
    site_record = sites[site]
    if not isinstance(site_record, Mapping):
        raise ExtractionError(f"Invalid structure site {site}")
    species = site_record.get("species")
    if not isinstance(species, list) or len(species) != 1 or not isinstance(species[0], Mapping):
        raise ExtractionError(f"Disordered or invalid species at source site {site}")
    element = species[0].get("element")
    if not isinstance(element, str) or not ELEMENT_RE.fullmatch(element):
        raise ExtractionError(f"Invalid element at source site {site}: {element!r}")
    return element, material, site


def raw_spectrum_table(record: Mapping[str, object]) -> np.ndarray:
    spectrum = record.get("spectrum")
    if isinstance(spectrum, Mapping):
        x = spectrum.get("x", spectrum.get("energy"))
        y = spectrum.get("y", spectrum.get("intensity"))
        if x is None or y is None:
            raise ExtractionError("Spectrum object has no energy and intensity arrays")
        if "x" in spectrum and "energy" in spectrum and spectrum["x"] != spectrum["energy"]:
            raise ExtractionError("Spectrum has conflicting x and energy arrays")
        if "y" in spectrum and "intensity" in spectrum and spectrum["y"] != spectrum["intensity"]:
            raise ExtractionError("Spectrum has conflicting y and intensity arrays")
        table = np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    else:
        table = np.asarray(spectrum, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] < 2:
        raise ExtractionError(f"Unexpected raw spectrum table shape: {table.shape}")
    if not np.isfinite(table).all():
        raise ExtractionError("Raw spectrum contains non-finite values")
    return table


def geometry_signature(structure: Mapping[str, object]) -> bytes:
    """Canonicalize physical coordinates enough to collapse rounding-only copies."""
    lattice = structure.get("lattice")
    if not isinstance(lattice, Mapping):
        raise ExtractionError("Structure has no lattice")
    matrix = np.asarray(lattice.get("matrix"), dtype=np.float64)
    sites = structure.get("sites")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not isinstance(sites, list):
        raise ExtractionError("Invalid source structure geometry")
    geometry: dict[str, object] = {"lattice": np.round(matrix, 6).tolist(), "sites": []}
    normalized_sites: list[dict[str, object]] = []
    for item in sites:
        if not isinstance(item, Mapping):
            raise ExtractionError("Invalid source structure site")
        species = item.get("species")
        if not isinstance(species, list):
            raise ExtractionError("Invalid site species")
        fractional = item.get("abc")
        if fractional is None:
            fractional = item.get("fractional_coordinates")
        if fractional is None:
            raise ExtractionError("Source structure site has no fractional coordinates")
        normalized_sites.append(
            {
                "species": [
                    (entry.get("element"), round(float(entry.get("occu", 1.0)), 6))
                    for entry in species
                    if isinstance(entry, Mapping)
                ],
                "frac": np.round(np.asarray(fractional, dtype=np.float64), 6).tolist(),
            }
        )
    geometry["sites"] = normalized_sites
    return json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")


def duplicate_signature(record: Mapping[str, object], table: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(table, dtype="<f8").tobytes(order="C"))
    digest.update(geometry_signature(structure_for(record)))
    digest.update(
        json.dumps(
            {
                "edge": record.get("edge"),
                "input_parameters": record.get("input_parameters"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def poscar_text(structure: Mapping[str, object], material: str, site: int) -> str:
    lattice_obj = structure.get("lattice")
    if not isinstance(lattice_obj, Mapping):
        raise ExtractionError("Structure has no lattice matrix")
    lattice = np.asarray(lattice_obj.get("matrix"), dtype=np.float64)
    sites = structure.get("sites")
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or not isinstance(sites, list):
        raise ExtractionError("Invalid structure lattice or sites")

    symbols: list[str] = []
    coordinates: list[np.ndarray] = []
    for index, site_record in enumerate(sites):
        if not isinstance(site_record, Mapping):
            raise ExtractionError(f"Invalid structure site {index}")
        species = site_record.get("species")
        if not isinstance(species, list) or len(species) != 1 or not isinstance(species[0], Mapping):
            raise ExtractionError(f"Cannot serialize disordered structure site {index} as POSCAR")
        symbol = species[0].get("element")
        occupancy = float(species[0].get("occu", 1.0))
        if not isinstance(symbol, str) or not ELEMENT_RE.fullmatch(symbol) or not math.isclose(occupancy, 1.0):
            raise ExtractionError(f"Unsupported POSCAR species at site {index}")
        frac = site_record.get("abc", site_record.get("fractional_coordinates"))
        if frac is None:
            raise ExtractionError(f"No fractional coordinates at site {index}")
        coordinate = np.asarray(frac, dtype=np.float64)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise ExtractionError(f"Invalid fractional coordinates at site {index}")
        symbols.append(symbol)
        coordinates.append(coordinate)

    run_symbols: list[str] = []
    run_counts: list[int] = []
    for symbol in symbols:
        if run_symbols and run_symbols[-1] == symbol:
            run_counts[-1] += 1
        else:
            run_symbols.append(symbol)
            run_counts.append(1)
    lines = [
        f"{material} | Figshare {ARTICLE_ID} | source site order | absorber {site}",
        "1.0",
        *["  " + "  ".join(f"{value:.16g}" for value in row) for row in lattice],
        "  " + "  ".join(run_symbols),
        "  " + "  ".join(str(value) for value in run_counts),
        "Direct",
        *["  " + "  ".join(f"{value:.16g}" for value in xyz) for xyz in coordinates],
    ]
    return "\n".join(lines) + "\n"


def write_candidate(directory: Path, record: Mapping[str, object], table: np.ndarray, key: Key) -> bool:
    """Write one source record; return whether a POSCAR could be represented."""
    directory.mkdir(parents=True, exist_ok=True)
    structure = structure_for(record)
    (directory / "figshare_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "structure.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "spectrum_raw.dat").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Numeric spectrum table as stored in Figshare article {ARTICLE_ID}.\n")
        handle.write("# Column meanings follow the source record. This is not an original xmu.dat.\n")
        np.savetxt(handle, table, fmt="%.12e")
    differences = np.diff(table[:, 0])
    nonmonotonic_count = int(np.count_nonzero(differences <= 0))
    (directory / "spectrum_raw_metadata.json").write_text(
        json.dumps(
            {
                "raw_spectrum_energy_monotonic": nonmonotonic_count == 0,
                "raw_spectrum_energy_non_monotonic_count": nonmonotonic_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        text = poscar_text(structure, key[1], key[2])
    except (ExtractionError, TypeError, ValueError) as exc:
        return False
    (directory / "POSCAR").write_text(text, encoding="utf-8", newline="\n")
    return True


def source_records(archive_path: Path) -> Iterator[tuple[int, Mapping[str, object]]]:
    """Yield records from Figshare's JSONL file without decompressing it to disk."""
    def from_stream(stream: BinaryIO) -> Iterator[tuple[int, Mapping[str, object]]]:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExtractionError(f"Invalid JSON in xas.json line {line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ExtractionError(f"Expected a JSON record at xas.json line {line_number}")
            yield line_number, value

    if archive_path.suffix.lower() == ".json":
        with archive_path.open("rb") as stream:
            yield from from_stream(stream)
        return
    if archive_path.name.lower().endswith(".json.gz"):
        with gzip.open(archive_path, "rb") as stream:
            yield from from_stream(stream)
        return
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or PurePosixPath(member.name).name != "xas.json":
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ExtractionError("Could not read xas.json from the Figshare tarball")
                with stream:
                    yield from from_stream(stream)
                return
    except tarfile.TarError as exc:
        raise ExtractionError(f"Could not read {archive_path} as a tar.gz archive: {exc}") from exc
    raise ExtractionError(f"No xas.json member found in {archive_path}")


def write_site_manifests(
    output_root: Path,
    keys: Sequence[Key],
    all_counts: Mapping[Key, int],
    xanes_counts: Mapping[Key, int],
    first_lines: Mapping[Key, int],
    poscar_errors: Sequence[tuple[Key, int]],
) -> dict[str, int]:
    manifest_path = output_root / "manifest.csv"
    pre_manifest_path = output_root / "manifest_pre_cleanup.csv"
    ambiguous_path = output_root / "ambiguous_keys.csv"
    pre_ambiguous_path = output_root / "ambiguous_keys_pre_cleanup.csv"
    missing_path = output_root / "missing_keys.csv"
    manifest_columns = MANIFEST_FIELDS
    pre_manifest_fields = tuple(field for field in MANIFEST_FIELDS if field != "raw_candidate_count_before_cleanup")
    ambiguous_fields = (
        "key",
        "element",
        "material_id",
        "site",
        "raw_candidate_count",
        "first_source_line",
        "status",
        "site_directory",
    )
    pre_ambiguous_fields = (*ambiguous_fields, "candidate_directories")
    missing_fields = (
        *ambiguous_fields,
        "candidate_directories",
    )
    counts = {"matched": 0, "ambiguous": 0, "missing": 0}
    with (
        manifest_path.open("w", newline="", encoding="utf-8") as manifest_handle,
        pre_manifest_path.open("w", newline="", encoding="utf-8") as pre_manifest_handle,
        ambiguous_path.open("w", newline="", encoding="utf-8") as ambiguous_handle,
        pre_ambiguous_path.open("w", newline="", encoding="utf-8") as pre_ambiguous_handle,
        missing_path.open("w", newline="", encoding="utf-8") as missing_handle,
    ):
        manifest_writer = csv.DictWriter(manifest_handle, fieldnames=manifest_columns)
        pre_manifest_writer = csv.DictWriter(pre_manifest_handle, fieldnames=pre_manifest_fields)
        ambiguous_writer = csv.DictWriter(ambiguous_handle, fieldnames=ambiguous_fields)
        pre_ambiguous_writer = csv.DictWriter(pre_ambiguous_handle, fieldnames=pre_ambiguous_fields)
        missing_writer = csv.DictWriter(missing_handle, fieldnames=missing_fields)
        manifest_writer.writeheader()
        pre_manifest_writer.writeheader()
        ambiguous_writer.writeheader()
        pre_ambiguous_writer.writeheader()
        missing_writer.writeheader()
        for row_index, key in enumerate(keys):
            element, material, site = key
            count = xanes_counts.get(key, 0)
            before_count = all_counts.get(key, 0)
            first_line = first_lines.get(key, 0)
            site_rel = PurePosixPath("FEFF", element, material, "FEFF-XANES", f"{site:03d}_{element}")
            status = "missing" if count == 0 else "matched" if count == 1 else "ambiguous"
            counts[status] += 1
            manifest_writer.writerow(
                {
                    "row_index": row_index,
                    "key": key_text(key),
                    "element": element,
                    "material_id": material,
                    "site": site,
                    # Splits are deliberately exported by the later split tool.
                    "split": "",
                    "raw_candidate_count": count,
                    "first_source_line": first_line,
                    "status": status,
                    "site_directory": str(site_rel),
                    "raw_candidate_count_before_cleanup": before_count,
                }
            )
            before_status = "missing" if before_count == 0 else "matched" if before_count == 1 else "ambiguous"
            pre_row = {
                "row_index": row_index,
                "key": key_text(key),
                "element": element,
                "material_id": material,
                "site": site,
                "split": "",
                "raw_candidate_count": before_count,
                "first_source_line": first_line,
                "status": before_status,
                "site_directory": str(site_rel),
            }
            pre_manifest_writer.writerow(pre_row)
            if before_count > 1:
                pre_ambiguous_writer.writerow(
                    {
                        "key": key_text(key),
                        "element": element,
                        "material_id": material,
                        "site": site,
                        "raw_candidate_count": before_count,
                        "first_source_line": first_line,
                        "status": before_status,
                        "site_directory": str(site_rel),
                        "candidate_directories": "",
                    }
                )
            if count > 1:
                ambiguous_writer.writerow(
                    {
                        "key": key_text(key),
                        "element": element,
                        "material_id": material,
                        "site": site,
                        "raw_candidate_count": count,
                        "first_source_line": first_line,
                        "status": status,
                        "site_directory": str(site_rel),
                    }
                )
            elif count == 0:
                missing_writer.writerow(
                    {
                    "key": key_text(key),
                    "element": element,
                    "material_id": material,
                    "site": site,
                    "raw_candidate_count": count,
                    "first_source_line": first_line,
                    "status": status,
                    "site_directory": str(site_rel),
                    "candidate_directories": "",
                    }
                )

    with (output_root / "record_cleanup_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "key",
                "site_directory",
                "action",
                "removed_candidate_count",
                "retained_record_id",
                "retained_record_type",
                "promoted_files",
                "reason",
                "error",
            )
        )
    with (output_root / "poscar_cleanup_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "deleted_or_checked_top_level_poscar",
                "matching_site_poscar",
                "sha256",
                "bytes",
                "action",
            )
        )
    return counts


def prepare_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ExtractionError(
            f"Output directory is not empty: {output_root}. Choose a fresh directory; "
            "the script never replaces an existing extraction."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "IN_PROGRESS").write_text("Extraction has not completed.\n", encoding="utf-8")


def run_extraction(
    archive_path: Path,
    output_root: Path,
    archive_md5: str | None,
) -> dict[str, object]:
    prepare_output(output_root)

    all_counts: dict[Key, int] = {}
    xanes_counts: dict[Key, int] = {}
    first_lines: dict[Key, int] = {}
    poscar_errors: list[tuple[Key, int]] = []
    malformed_count = 0
    malformed_samples: list[dict[str, object]] = []
    exact_duplicates_removed = 0
    source_record_count = 0
    xanes_record_count = 0
    nonmonotonic_count = 0
    signatures: dict[Key, set[str]] = {}

    for line_number, record in source_records(archive_path):
        source_record_count += 1
        try:
            key = source_key(record)
        except ExtractionError as exc:
            malformed_count += 1
            if len(malformed_samples) < 20:
                malformed_samples.append({"source_line": line_number, "error": str(exc)})
            continue
        all_counts[key] = all_counts.get(key, 0) + 1
        first_lines.setdefault(key, line_number)
        record_type = str(record.get("spectrum_type", "")).strip().upper()
        if record_type != "XANES":
            continue
        xanes_record_count += 1
        try:
            table = raw_spectrum_table(record)
            signature = duplicate_signature(record, table)
        except (ExtractionError, TypeError, ValueError) as exc:
            malformed_count += 1
            if len(malformed_samples) < 20:
                malformed_samples.append({"source_line": line_number, "key": key_text(key), "error": str(exc)})
            continue
        seen_signatures = signatures.setdefault(key, set())
        if signature in seen_signatures:
            exact_duplicates_removed += 1
            continue
        seen_signatures.add(signature)

        candidate_index = xanes_counts.get(key, 0)
        if candidate_index == 0:
            candidate_directory = safe_site_directory(output_root, key) / "candidates" / "000"
        else:
            candidate_directory = (
                safe_site_directory(output_root, key)
                / "candidates"
                / f"{candidate_index:03d}"
            )
        try:
            poscar_ok = write_candidate(candidate_directory, record, table, key)
            if not poscar_ok:
                poscar_errors.append((key, line_number))
            nonmonotonic_count += int(np.count_nonzero(np.diff(table[:, 0]) <= 0) > 0)
        except (OSError, ExtractionError, TypeError, ValueError) as exc:
            malformed_count += 1
            if len(malformed_samples) < 20:
                malformed_samples.append({"source_line": line_number, "key": key_text(key), "error": str(exc)})
            continue
        xanes_counts[key] = candidate_index + 1

        if source_record_count % 25_000 == 0:
            print(
                f"Scanned {source_record_count:,} Figshare records; "
                f"retained {sum(xanes_counts.values()):,} unique XANES records",
                flush=True,
            )

    # A one-candidate site is stored directly in the site directory. Ambiguous
    # sites retain all differing source records under candidates/000, 001, ... .
    for key, count in xanes_counts.items():
        if count != 1:
            continue
        site_directory = safe_site_directory(output_root, key)
        candidate_directory = site_directory / "candidates" / "000"
        for filename in SITE_FILES:
            source = candidate_directory / filename
            if source.exists():
                source.replace(site_directory / filename)
        candidate_directory.rmdir()
        (site_directory / "candidates").rmdir()

    keys = sorted(xanes_counts)
    status_counts = write_site_manifests(
        output_root, keys, all_counts, xanes_counts, first_lines, poscar_errors
    )

    report: dict[str, object] = {
        "figshare_article_id": ARTICLE_ID,
        "figshare_archive_file_id": ARCHIVE_FILE_ID,
        "source_archive": str(archive_path.resolve()),
        "source_archive_md5": archive_md5,
        "row_count": len(keys),
        "source_record_count": source_record_count,
        "matching_source_record_count_before_cleanup": int(sum(all_counts.values())),
        "matching_xanes_record_count": xanes_record_count,
        "elnes_or_other_type_record_count_removed": int(sum(all_counts.values()) - xanes_record_count),
        "exact_duplicate_xanes_records_removed": exact_duplicates_removed,
        "unique_xanes_candidate_count": int(sum(xanes_counts.values())),
        "source_scan_ambiguous_key_count_before_cleanup": sum(value > 1 for value in all_counts.values()),
        "matched_key_count": status_counts["matched"],
        "ambiguous_key_count": status_counts["ambiguous"],
        "missing_key_count": status_counts["missing"],
        "poscar_not_written_count": len(poscar_errors),
        "malformed_or_unwritable_record_count": malformed_count,
        "malformed_or_unwritable_record_samples": malformed_samples,
        "raw_spectrum_nonmonotonic_record_count": nonmonotonic_count,
        "spectrum_type_used": "XANES only; ELNES records are counted but not extracted",
        "raw_spectrum_policy": "Original source table copied as numeric values; no alignment, interpolation, or rescaling",
        "final_spectrum_policy": "The Figshare article does not contain the separate final-target NPZ; only raw source spectra are extracted",
        "split_policy": "No split assignment; manifest split column is blank for the separate split-export step",
        "site_directory_policy": "Original Figshare absorbing site index; no cross-source reindexing",
        "status_counts": status_counts,
    }
    pre_cleanup_report = {
        "figshare_article_id": ARTICLE_ID,
        "figshare_archive_file_id": ARCHIVE_FILE_ID,
        "source_archive": str(archive_path.resolve()),
        "row_count": len(keys),
        "source_record_count": source_record_count,
        "matching_source_record_count_before_cleanup": int(sum(all_counts.values())),
        "source_scan_ambiguous_key_count_before_cleanup": sum(value > 1 for value in all_counts.values()),
        "split_policy": "No split assignment; manifest split column is blank for the separate split-export step",
    }
    (output_root / "report_pre_cleanup.json").write_text(
        json.dumps(pre_cleanup_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# Figshare extraction\n\n"
        f"Figshare article {ARTICLE_ID}, archive file {ARCHIVE_FILE_ID} (`xas.json.tgz`).\n\n"
        "The `FEFF/<element>/<material>/FEFF-XANES/<site>_<element>/` hierarchy "
        "preserves Figshare's original absorbing-site indices. A single XANES "
        "record is stored directly in the site directory; multiple differing "
        "records are retained under `candidates/`. ELNES records are excluded.\n\n"
        "Each source candidate contains `figshare_record.json`, `structure.json`, "
        "`POSCAR` when the structure can be represented as an ordered POSCAR, "
        "`spectrum_raw.dat`, and `spectrum_raw_metadata.json`. The Figshare article "
        "does not contain the separate final-target NPZ. Raw spectra are not "
        "energy-aligned, interpolated, normalized, or repaired.\n\n"
        "`manifest.csv` keeps the desktop column layout; `split` is blank because "
        "splits are assigned separately. See `report.json`, `ambiguous_keys.csv`, "
        "`missing_keys.csv`, and `poscar_cleanup_log.csv` for extraction status.\n",
        encoding="utf-8",
    )
    (output_root / "IN_PROGRESS").unlink()
    (output_root / "COMPLETE").write_text("Extraction completed successfully.\n", encoding="utf-8")
    return report


def verify_archive_md5(path: Path) -> str:
    # Figshare publishes this checksum as an archive-integrity check, not a
    # cryptographic security primitive; usedforsecurity=False also permits FIPS.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != EXPECTED_ARCHIVE_MD5:
        raise ExtractionError(
            f"Figshare archive MD5 mismatch: expected {EXPECTED_ARCHIVE_MD5}, got {actual}"
        )
    return actual


def download_archive(path: Path) -> str:
    """Download once to a temporary sibling and verify the published checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        actual = verify_archive_md5(path)
        print(f"Using verified Figshare archive: {path}", flush=True)
        return actual
    partial = path.with_name(path.name + ".part")
    digest = hashlib.md5(usedforsecurity=False)
    print(f"Downloading Figshare archive to {path} (5.56 GB)...", flush=True)
    try:
        with urlopen(ARCHIVE_URL, timeout=120) as response, partial.open("wb") as output:
            downloaded = 0
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded % (1024 * 1024 * 1024) < len(chunk):
                    print(f"Downloaded {downloaded / (1024 ** 3):.1f} GiB", flush=True)
        actual = digest.hexdigest()
        if actual != EXPECTED_ARCHIVE_MD5:
            raise ExtractionError(
                f"Figshare archive MD5 mismatch: expected {EXPECTED_ARCHIVE_MD5}, got {actual}"
            )
        partial.replace(path)
        return actual
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def default_paths() -> tuple[Path, Path]:
    data_root = Path(__file__).resolve().parents[2] / "OmniXAS_data"
    return data_root / "downloads" / "xas.json.tgz", data_root / "figshare"


def main() -> int:
    archive_path, output_root = default_paths()
    try:
        archive_md5 = download_archive(archive_path)
        report = run_extraction(archive_path, output_root, archive_md5)
    except (ExtractionError, OSError, tarfile.TarError, TimeoutError) as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 2
    print(
        "Extraction complete: "
        f"rows={report['row_count']:,}, matched={report['matched_key_count']:,}, "
        f"ambiguous={report['ambiguous_key_count']:,}, missing={report['missing_key_count']:,}"
    )
    print(f"Output: {output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
