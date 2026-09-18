#!/usr/bin/env python3
"""Extract curated AnionXAS structures and spectra from the Figshare archive.

The clean AnionXAS NPZ pair defines the accepted ``(element, material_id, site)``
keys and target values. The Figshare archive supplies the source structure and
raw spectral table for each accepted key. The program streams ``xas.json.tgz``
without writing the approximately 33 GB decompressed JSON file.

The program does not create ``feff.inp`` or ``feff.out``. Figshare does not
publish those original FEFF files.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from itertools import islice
import random
import re
import shutil
import sys
import tarfile
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence, TextIO

import numpy as np


FIGSHARE_ARTICLE_ID = 5678998
FIGSHARE_FILE_ID = 9932248
FIGSHARE_ARCHIVE_MD5 = "e866677ebb9270aeb2e15c725bef7e05"
SOURCE_POINTS = 200
TARGET_POINTS = 141
ENERGY_MIN_EV = 0.0
ENERGY_MAX_EV = 35.0
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
ELEMENT_PATTERN = re.compile(r"^[A-Z][a-z]?$")
SAFE_MATERIAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ExtractionError(RuntimeError):
    """Raised when input data cannot produce a valid curated dataset."""


@dataclass(frozen=True, order=True)
class CuratedKey:
    """One absorber-site identity."""

    element: str
    material_id: str
    site: int

    @property
    def text(self) -> str:
        return str((self.element, self.material_id, self.site))

    @property
    def site_directory_name(self) -> str:
        return f"{self.site:03d}_{self.element}"


@dataclass
class ScanResult:
    """Summary from the streamed Figshare scan."""

    records_read: int
    matching_candidates: int
    parse_error_count: int
    record_error_count: int
    parse_error_samples: list[str]
    record_error_samples: list[str]
    structure_conflicts: set[tuple[str, str]]


def parse_curated_key(text: str) -> CuratedKey:
    """Parse and validate one NPZ member key."""
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ExtractionError(f"Invalid curated key {text!r}") from exc
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ExtractionError(f"Expected a three-item curated key, got {text!r}")

    element, material_id, site = value
    element = str(element)
    material_id = str(material_id)
    try:
        site = int(site)
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"Invalid site in curated key {text!r}") from exc

    if not ELEMENT_PATTERN.fullmatch(element):
        raise ExtractionError(f"Invalid element in curated key {text!r}")
    if not SAFE_MATERIAL_PATTERN.fullmatch(material_id):
        raise ExtractionError(f"Unsafe material ID in curated key {text!r}")
    if site < 0:
        raise ExtractionError(f"Negative site in curated key {text!r}")
    return CuratedKey(element, material_id, site)


def load_spectral_key_archive(
    spectral_path: Path,
) -> tuple[list[CuratedKey], dict[CuratedKey, str]]:
    """Load and validate curated keys from the authoritative spectra archive."""
    try:
        with np.load(spectral_path, allow_pickle=False) as spectral_archive:
            spectral_names = list(spectral_archive.files)
            if not spectral_names:
                raise ExtractionError(f"Spectrum archive is empty: {spectral_path}")
    except (OSError, ValueError) as exc:
        raise ExtractionError(f"Could not read the curated spectrum NPZ: {exc}") from exc

    name_by_key: dict[CuratedKey, str] = {}
    for name in spectral_names:
        key = parse_curated_key(name)
        if key in name_by_key:
            raise ExtractionError(
                f"Multiple NPZ member names resolve to curated key {key.text}"
            )
        name_by_key[key] = name
    return sorted(name_by_key), name_by_key


def load_key_file(path: Path) -> list[CuratedKey]:
    """Load a compact CSV keep-list without requiring any target arrays."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"key", "element", "material_id", "site"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise ExtractionError(
                    f"Key file must have exactly these columns: {sorted(required)}"
                )
            keys: list[CuratedKey] = []
            for row_number, row in enumerate(reader, 2):
                key = parse_curated_key(row["key"])
                if (row["element"], row["material_id"], int(row["site"])) != (
                    key.element, key.material_id, key.site
                ):
                    raise ExtractionError(f"Key file identity mismatch at row {row_number}")
                keys.append(key)
    except (OSError, KeyError, ValueError) as exc:
        raise ExtractionError(f"Could not read key file: {exc}") from exc
    if not keys:
        raise ExtractionError(f"Key file is empty: {path}")
    if len(set(keys)) != len(keys):
        raise ExtractionError("Key file contains duplicate keys")
    return sorted(keys)


def write_key_file(path: Path, keys: Sequence[CuratedKey]) -> None:
    """Write a stable, human-readable keep-list (gzip when suffix is .gz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "element", "material_id", "site"))
        for key in keys:
            writer.writerow((key.text, key.element, key.material_id, key.site))


def export_clean_key_file(spectral_path: Path, output_path: Path) -> dict[str, int]:
    """Export keys whose clean 200-point targets contain no negative values."""
    source_keys, names = load_spectral_key_archive(spectral_path)
    retained: list[CuratedKey] = []
    excluded = 0
    with np.load(spectral_path, allow_pickle=False) as spectra:
        for key in source_keys:
            values = np.asarray(spectra[names[key]], dtype=np.float64)
            if values.shape != (SOURCE_POINTS,) or not np.isfinite(values).all():
                raise ExtractionError(f"Invalid clean target for {key.text}")
            if float(values.min()) < 0:
                excluded += 1
            else:
                retained.append(key)
    write_key_file(output_path, retained)
    return {"source_count": len(source_keys), "key_count": len(retained), "excluded_negative_count": excluded}


def load_aligned_key_archives(
    feature_path: Path, spectral_path: Path
) -> tuple[list[CuratedKey], dict[CuratedKey, str]]:
    """Load keys from two NPZ files and require exact key-set alignment."""
    try:
        with np.load(feature_path, allow_pickle=False) as feature_archive:
            feature_names = list(feature_archive.files)
            if not feature_names:
                raise ExtractionError(f"Feature archive is empty: {feature_path}")
            for feature_name in feature_names:
                sample = np.asarray(feature_archive[feature_name])
                if sample.shape != (64,):
                    raise ExtractionError(
                        f"Expected 64 feature values for {feature_name!r}, "
                        f"got {sample.shape} in {feature_path}"
                    )
    except (OSError, ValueError) as exc:
        raise ExtractionError(f"Could not read the curated feature NPZ: {exc}") from exc

    keys, name_by_key = load_spectral_key_archive(spectral_path)
    spectral_set = set(name_by_key.values())
    feature_set = set(feature_names)
    if feature_set != spectral_set:
        raise ExtractionError(
            "Curated NPZ key sets differ: "
            f"feature-only={len(feature_set - spectral_set)}, "
            f"spectrum-only={len(spectral_set - feature_set)}"
        )
    return keys, name_by_key


def resample_target(values: np.ndarray) -> np.ndarray:
    """Resample one curated 35 eV target to the 141-point OmniXAS spacing."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (SOURCE_POINTS,):
        raise ExtractionError(
            f"Expected a {SOURCE_POINTS}-point curated target, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ExtractionError("Curated target contains NaN or infinite values")
    minimum = float(values.min())
    if minimum < 0:
        raise ExtractionError(f"Curated target contains a negative value: {minimum}")

    source_grid = np.linspace(ENERGY_MIN_EV, ENERGY_MAX_EV, SOURCE_POINTS)
    target_grid = np.linspace(ENERGY_MIN_EV, ENERGY_MAX_EV, TARGET_POINTS)
    return np.interp(target_grid, source_grid, values).astype(np.float32)


def assign_material_splits(
    keys: Sequence[CuratedKey], seed: int
) -> tuple[dict[str, int], tuple[int, int, int]]:
    """Assign each material globally while keeping row counts near 80/10/10."""
    row_counts: dict[str, int] = {}
    for key in keys:
        row_counts[key.material_id] = row_counts.get(key.material_id, 0) + 1

    rng = random.Random(seed)
    tie_breakers = {material_id: rng.random() for material_id in row_counts}
    materials = sorted(
        row_counts,
        key=lambda material_id: (-row_counts[material_id], tie_breakers[material_id]),
    )

    assigned_rows = [0, 0, 0]
    assignment: dict[str, int] = {}
    for material_id in materials:
        split_code = min(
            range(3),
            key=lambda code: (
                assigned_rows[code] / SPLIT_FRACTIONS[code],
                code,
            ),
        )
        assignment[material_id] = split_code
        assigned_rows[split_code] += row_counts[material_id]

    return assignment, tuple(assigned_rows)


def file_digest(path: Path, algorithm: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a file digest without loading the file into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _first_value(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _one_consistent_value(name: str, values: Sequence[object]) -> object:
    """Return one value and reject conflicting copies of an identity field."""
    present = [value for value in values if value is not None]
    if not present:
        raise ExtractionError(f"Record has no {name}")
    if any(value != present[0] for value in present[1:]):
        raise ExtractionError(f"Record has conflicting {name} values: {present!r}")
    return present[0]


def record_material_site(record: Mapping[str, object]) -> tuple[str, int]:
    """Read and cross-check the material ID and absorber index."""
    metadata_value = record.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    spectrum_value = record.get("spectrum")
    spectrum = spectrum_value if isinstance(spectrum_value, Mapping) else {}

    material_value = _one_consistent_value(
        "material ID",
        (
            metadata.get("mp_id"),
            metadata.get("material_id"),
            record.get("material_id"),
            spectrum.get("material_id"),
        ),
    )
    if not isinstance(material_value, str):
        raise ExtractionError(f"Material ID is not text: {material_value!r}")
    if not SAFE_MATERIAL_PATTERN.fullmatch(material_value):
        raise ExtractionError(f"Unsafe material ID: {material_value!r}")

    raw_site_values = [
        value
        for value in (
            metadata.get("absorbing_atom_index"),
            record.get("absorbing_atom"),
            spectrum.get("absorbing_index"),
            spectrum.get("absorbing_atom"),
        )
        if value is not None
    ]
    if not raw_site_values:
        raise ExtractionError("Record has no absorber index")
    normalized_sites: list[int] = []
    for value in raw_site_values:
        if isinstance(value, bool):
            raise ExtractionError(f"Invalid absorber index {value!r}")
        if isinstance(value, int):
            normalized_sites.append(value)
        elif isinstance(value, str) and value.isdigit():
            normalized_sites.append(int(value))
        else:
            raise ExtractionError(f"Invalid absorber index {value!r}")
    if any(value != normalized_sites[0] for value in normalized_sites[1:]):
        raise ExtractionError(
            f"Record has conflicting absorber index values: {raw_site_values!r}"
        )
    site_number = normalized_sites[0]
    if site_number < 0:
        raise ExtractionError(f"Negative absorber index {site_number}")
    return material_value, site_number


def record_structure(record: Mapping[str, object]) -> Mapping[str, object]:
    """Read a Pymatgen structure dictionary from one Figshare record."""
    structure = record.get("structure")
    spectrum = record.get("spectrum")
    if structure is None and isinstance(spectrum, Mapping):
        structure = spectrum.get("structure")
    if not isinstance(structure, Mapping):
        raise ExtractionError("Record has no structure dictionary")
    return structure


def site_element(structure: Mapping[str, object], site: int) -> str:
    """Read the element at one structure index."""
    sites = structure.get("sites")
    if not isinstance(sites, list) or not 0 <= site < len(sites):
        raise ExtractionError(f"Absorber index {site} is outside the structure")
    site_data = sites[site]
    if not isinstance(site_data, Mapping):
        raise ExtractionError(f"Structure site {site} is not an object")

    label = site_data.get("label")
    if isinstance(label, str):
        match = re.match(r"^([A-Z][a-z]?)", label)
        if match:
            return match.group(1)

    species = site_data.get("species")
    if isinstance(species, list) and species and isinstance(species[0], Mapping):
        element = species[0].get("element")
        if isinstance(element, str) and ELEMENT_PATTERN.fullmatch(element):
            return element
    raise ExtractionError(f"Could not identify the element at structure site {site}")


def record_spectrum_table(record: Mapping[str, object]) -> np.ndarray:
    """Read and validate the complete numeric spectrum table from Figshare."""
    spectrum = record.get("spectrum")
    if isinstance(spectrum, Mapping):
        x_value = spectrum.get("x")
        energy_value = spectrum.get("energy")
        y_value = spectrum.get("y")
        intensity_value = spectrum.get("intensity")
        if x_value is not None and energy_value is not None and not np.array_equal(
            np.asarray(x_value), np.asarray(energy_value)
        ):
            raise ExtractionError("Spectrum has conflicting x and energy arrays")
        if y_value is not None and intensity_value is not None and not np.array_equal(
            np.asarray(y_value), np.asarray(intensity_value)
        ):
            raise ExtractionError("Spectrum has conflicting y and intensity arrays")
        x = _first_value(x_value, energy_value)
        y = _first_value(y_value, intensity_value)
        if x is None or y is None:
            raise ExtractionError("Spectrum object has no x/y arrays")
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        if x_array.ndim != 1 or y_array.ndim != 1:
            raise ExtractionError("Spectrum x/y values must be one-dimensional")
        if x_array.shape != y_array.shape:
            raise ExtractionError(
                f"Spectrum x/y lengths differ: {x_array.shape} and {y_array.shape}"
            )
        table = np.column_stack((x_array, y_array))
    else:
        table = np.asarray(spectrum, dtype=np.float64)

    table = np.asarray(table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ExtractionError(f"Unexpected raw spectrum shape {table.shape}")
    if not np.isfinite(table).all():
        raise ExtractionError("Raw spectrum contains NaN or infinite values")
    if not np.all(np.diff(table[:, 0]) > 0):
        raise ExtractionError("Raw spectrum energy values are not strictly increasing")
    return table


def canonical_structure_digest(structure: Mapping[str, object]) -> str:
    """Hash one structure dictionary with stable JSON formatting."""
    data = json.dumps(structure, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _ordered_site_data(
    structure: Mapping[str, object],
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    lattice_value = structure.get("lattice")
    if not isinstance(lattice_value, Mapping):
        raise ExtractionError("Structure has no lattice object")
    lattice = np.asarray(lattice_value.get("matrix"), dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ExtractionError(f"Unexpected lattice shape {lattice.shape}")

    site_values = structure.get("sites")
    if not isinstance(site_values, list) or not site_values:
        raise ExtractionError("Structure has no sites")

    symbols: list[str] = []
    coordinates: list[np.ndarray] = []
    for index, value in enumerate(site_values):
        if not isinstance(value, Mapping):
            raise ExtractionError(f"Structure site {index} is not an object")
        species = value.get("species")
        if not isinstance(species, list) or len(species) != 1:
            raise ExtractionError(
                f"Structure site {index} is disordered and cannot be written as POSCAR"
            )
        species_data = species[0]
        if not isinstance(species_data, Mapping):
            raise ExtractionError(f"Structure site {index} has invalid species data")
        symbol = species_data.get("element")
        occupancy = species_data.get("occu", 1.0)
        if not isinstance(symbol, str) or not ELEMENT_PATTERN.fullmatch(symbol):
            raise ExtractionError(f"Structure site {index} has no valid element")
        if not np.isclose(float(occupancy), 1.0):
            raise ExtractionError(
                f"Structure site {index} has occupancy {occupancy}, not 1"
            )
        abc = _first_value(value.get("abc"), value.get("frac_coords"))
        coordinate = np.asarray(abc, dtype=np.float64)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise ExtractionError(
                f"Structure site {index} has invalid fractional coordinates"
            )
        symbols.append(symbol)
        coordinates.append(coordinate)
    return lattice, symbols, coordinates


def structure_to_poscar(
    structure: Mapping[str, object], material_id: str, source_site: int
) -> str:
    """Serialize a structure without changing its site order.

    Repeated element labels are allowed in the POSCAR symbol line when species
    occur in separate runs. This keeps coordinate-line indices equal to the
    archived structure indices.
    """
    lattice, symbols, coordinates = _ordered_site_data(structure)
    run_symbols: list[str] = []
    run_counts: list[int] = []
    for symbol in symbols:
        if run_symbols and run_symbols[-1] == symbol:
            run_counts[-1] += 1
        else:
            run_symbols.append(symbol)
            run_counts.append(1)

    lines = [
        f"{material_id} | Figshare 5678998 | source site order | absorber {source_site}",
        "1.0",
    ]
    lines.extend("  " + "  ".join(f"{value:.16g}" for value in row) for row in lattice)
    lines.append("  " + "  ".join(run_symbols))
    lines.append("  " + "  ".join(str(value) for value in run_counts))
    lines.append("Direct")
    lines.extend(
        "  " + "  ".join(f"{value:.16g}" for value in coordinate)
        for coordinate in coordinates
    )
    return "\n".join(lines) + "\n"


def iter_json_lines(archive_path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield JSON records from JSON, gzip, or tar-gzip input."""
    suffixes = [suffix.lower() for suffix in archive_path.suffixes]
    if archive_path.suffix.lower() == ".json":
        binary_handle: BinaryIO = archive_path.open("rb")
        yield from _records_from_binary_stream(binary_handle)
        binary_handle.close()
        return

    if suffixes[-2:] == [".json", ".gz"]:
        with gzip.open(archive_path, "rb") as binary_handle:
            yield from _records_from_binary_stream(binary_handle)
        return

    try:
        with tarfile.open(archive_path, mode="r|gz") as tar:
            found = False
            for member in tar:
                if not member.isfile() or Path(member.name).name != "xas.json":
                    continue
                found = True
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ExtractionError("Could not read xas.json from the archive")
                yield from _records_from_binary_stream(extracted)
                break
            if not found:
                raise ExtractionError("The archive does not contain xas.json")
    except tarfile.TarError as exc:
        raise ExtractionError(f"Could not read {archive_path} as tar-gzip: {exc}") from exc


def _records_from_binary_stream(
    binary_handle: BinaryIO,
) -> Iterator[tuple[int, dict[str, object]]]:
    for line_number, raw_line in enumerate(binary_handle, start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExtractionError(
                f"Invalid UTF-8 at xas.json line {line_number}: {exc}"
            ) from exc
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"Invalid JSON at xas.json line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ExtractionError(
                f"Expected a JSON object at xas.json line {line_number}"
            )
        yield line_number, value


def site_directory(output_dir: Path, key: CuratedKey) -> Path:
    """Return the OmniXAS-style directory for one site."""
    return (
        output_dir
        / "extracted"
        / "FEFF"
        / key.element
        / key.material_id
        / "FEFF-XANES"
        / key.site_directory_name
    )


def load_curated_targets(
    keys: Sequence[CuratedKey],
    name_by_key: Mapping[CuratedKey, str],
    spectral_path: Path,
) -> tuple[list[CuratedKey], np.ndarray, list[dict[str, object]]]:
    """Load targets, excluding every spectrum with negative intensity."""
    retained_keys: list[CuratedKey] = []
    targets = np.empty((len(keys), TARGET_POINTS), dtype=np.float32)
    retained_count = 0
    excluded: list[dict[str, object]] = []

    with np.load(spectral_path, allow_pickle=False) as spectra:
        for index, key in enumerate(keys):
            values = np.asarray(spectra[name_by_key[key]], dtype=np.float64)
            if values.shape != (SOURCE_POINTS,):
                raise ExtractionError(
                    f"Expected a {SOURCE_POINTS}-point target for {key.text}, "
                    f"got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ExtractionError(f"Curated target is not finite for {key.text}")
            minimum = float(values.min())
            if minimum < 0:
                excluded.append({"key": key.text, "minimum": minimum})
            else:
                retained_keys.append(key)
                targets[retained_count] = resample_target(values)
                retained_count += 1
            if (index + 1) % 50_000 == 0:
                print(f"Validated {index + 1:,}/{len(keys):,} curated targets")

    if not retained_keys:
        raise ExtractionError("No nonnegative curated targets remain")
    return retained_keys, targets[:retained_count], excluded


def write_target_files(
    output_dir: Path,
    keys: Sequence[CuratedKey],
    targets: np.ndarray,
) -> None:
    """Write one two-column, 141-point target file per curated key."""
    if targets.shape != (len(keys), TARGET_POINTS):
        raise ExtractionError(
            f"Target matrix shape {targets.shape} does not match {len(keys)} keys"
        )
    energy = np.linspace(ENERGY_MIN_EV, ENERGY_MAX_EV, TARGET_POINTS)
    for index, (key, target) in enumerate(zip(keys, targets)):
        directory = site_directory(output_dir, key)
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            directory / "spectrum_141.dat",
            np.column_stack((energy, target)),
            fmt="%.8e",
            header=(
                "relative_energy_eV curated_anionxas_intensity\n"
                "141 points; 0.25 eV spacing; intensity scale unchanged"
            ),
        )
        if (index + 1) % 10_000 == 0:
            print(f"Wrote {index + 1:,}/{len(keys):,} curated target files")


def _write_raw_candidate(
    directory: Path,
    structure: Mapping[str, object],
    raw_spectrum: np.ndarray,
    key: CuratedKey,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "structure.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "POSCAR").write_text(
        structure_to_poscar(structure, key.material_id, key.site),
        encoding="utf-8",
    )
    np.savetxt(
        directory / "spectrum_raw.dat",
        raw_spectrum,
        fmt="%.12e",
        header=(
            "Numeric spectrum table as stored in Figshare article 5678998.\n"
            "Column meanings follow the source record. This is not an original xmu.dat."
        ),
    )


def scan_figshare_archive(
    archive_path: Path,
    output_dir: Path,
    keys: Sequence[CuratedKey],
    candidate_counts: np.ndarray,
    first_source_lines: np.ndarray,
    *,
    max_records: int | None = None,
) -> ScanResult:
    """Stream Figshare and write matching structures and raw spectra."""
    key_to_index = {key: index for index, key in enumerate(keys)}
    expected_elements: dict[tuple[str, int], set[str]] = {}
    for key in keys:
        expected_elements.setdefault((key.material_id, key.site), set()).add(key.element)

    material_structure_hash: dict[tuple[str, str], str | None] = {}
    structure_conflicts: set[tuple[str, str]] = set()
    parse_error_samples: list[str] = []
    record_error_samples: list[str] = []
    parse_error_count = 0
    record_error_count = 0
    records_read = 0
    matching_candidates = 0

    try:
        records: Iterable[tuple[int, dict[str, object]]] = iter_json_lines(archive_path)
        if max_records is not None:
            records = islice(records, max_records)
        for line_number, record in records:
            records_read += 1
            try:
                material_id, site = record_material_site(record)
            except ExtractionError as exc:
                parse_error_count += 1
                if len(parse_error_samples) < 20:
                    parse_error_samples.append(f"line {line_number}: {exc}")
                continue

            expected = expected_elements.get((material_id, site))
            if not expected:
                if records_read % 100_000 == 0:
                    print(
                        f"Scanned {records_read:,} Figshare records; "
                        f"matched {matching_candidates:,} candidates"
                    )
                continue

            try:
                structure = record_structure(record)
                element = site_element(structure, site)
                # Optional source labels are useful identity checks, but the
                # structure site remains authoritative when a source omits them.
                metadata = record.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                spectrum = record.get("spectrum")
                spectrum = spectrum if isinstance(spectrum, Mapping) else {}
                source_elements = [
                    value
                    for value in (
                        metadata.get("element"),
                        metadata.get("absorbing_element"),
                        record.get("element"),
                        record.get("absorbing_element"),
                        spectrum.get("element"),
                        spectrum.get("absorbing_element"),
                    )
                    if value is not None
                ]
                if any(value != element for value in source_elements):
                    raise ExtractionError(
                        f"Source element labels {source_elements!r} disagree "
                        f"with structure element {element!r}"
                    )
                key = CuratedKey(element, material_id, site)
                index = key_to_index.get(key)
                if index is None:
                    raise ExtractionError(
                        f"Structure element {element} does not match curated "
                        f"element(s) {sorted(expected)}"
                    )
                raw_spectrum = record_spectrum_table(record)
                candidate_number = int(candidate_counts[index])
                base_directory = site_directory(output_dir, key)
                if candidate_number == 1:
                    first_candidate_directory = base_directory / "candidates" / "000"
                    first_candidate_directory.mkdir(parents=True, exist_ok=True)
                    for name in ("POSCAR", "structure.json", "spectrum_raw.dat"):
                        (base_directory / name).replace(first_candidate_directory / name)
                candidate_directory = (
                    base_directory
                    if candidate_number == 0
                    else base_directory / "candidates" / f"{candidate_number:03d}"
                )
                _write_raw_candidate(
                    candidate_directory, structure, raw_spectrum, key
                )

                material_key = (element, material_id)
                structure_hash = canonical_structure_digest(structure)
                known_hash = material_structure_hash.get(material_key, "missing")
                material_directory = (
                    output_dir / "extracted" / "FEFF" / element / material_id
                )
                material_poscar = material_directory / "POSCAR"
                if known_hash == "missing":
                    material_structure_hash[material_key] = structure_hash
                    shutil.copyfile(candidate_directory / "POSCAR", material_poscar)
                elif known_hash is not None and known_hash != structure_hash:
                    material_structure_hash[material_key] = None
                    structure_conflicts.add(material_key)
                    material_poscar.unlink(missing_ok=True)

                candidate_counts[index] += 1
                if candidate_number == 0:
                    first_source_lines[index] = line_number
                matching_candidates += 1
            except (ExtractionError, OSError, TypeError, ValueError) as exc:
                record_error_count += 1
                if len(record_error_samples) < 20:
                    record_error_samples.append(
                        f"line {line_number}, {material_id}, site {site}: {exc}"
                    )

            if records_read % 100_000 == 0:
                print(
                    f"Scanned {records_read:,} Figshare records; "
                    f"matched {matching_candidates:,} candidates"
                )
    except ExtractionError as exc:
        parse_error_count += 1
        if len(parse_error_samples) < 20:
            parse_error_samples.append(str(exc))

    return ScanResult(
        records_read=records_read,
        matching_candidates=matching_candidates,
        parse_error_count=parse_error_count,
        record_error_count=record_error_count,
        parse_error_samples=parse_error_samples,
        record_error_samples=record_error_samples,
        structure_conflicts=structure_conflicts,
    )


def write_combined_target_archive(
    path: Path,
    keys: Sequence[CuratedKey],
    targets: np.ndarray,
    split_codes: np.ndarray,
) -> None:
    """Write compact aligned arrays without object or pickle data."""
    energy = np.linspace(
        ENERGY_MIN_EV, ENERGY_MAX_EV, TARGET_POINTS, dtype=np.float32
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            keys=np.asarray([key.text for key in keys]),
            elements=np.asarray([key.element for key in keys]),
            material_ids=np.asarray([key.material_id for key in keys]),
            sites=np.asarray([key.site for key in keys], dtype=np.int64),
            energies=energy,
            spectras=np.asarray(targets, dtype=np.float32),
            split_codes=np.asarray(split_codes, dtype=np.int8),
        )
    temporary_path.replace(path)


def load_target_package(
    path: Path,
) -> tuple[list[CuratedKey], np.ndarray, np.ndarray]:
    """Load and validate a portable, already-resampled target package."""
    try:
        with np.load(path, allow_pickle=False) as package:
            required = {"keys", "elements", "material_ids", "sites", "energies", "spectras", "split_codes"}
            missing = required - set(package.files)
            if missing:
                raise ExtractionError(f"Target package is missing keys: {sorted(missing)}")
            raw_keys = np.asarray(package["keys"])
            elements = np.asarray(package["elements"])
            materials = np.asarray(package["material_ids"])
            sites = np.asarray(package["sites"])
            energies = np.asarray(package["energies"], dtype=np.float64)
            targets = np.asarray(package["spectras"], dtype=np.float32)
            split_codes = np.asarray(package["split_codes"])
    except (OSError, ValueError) as exc:
        raise ExtractionError(f"Could not read target package: {exc}") from exc
    if raw_keys.ndim != 1 or elements.shape != raw_keys.shape or materials.shape != raw_keys.shape or sites.shape != raw_keys.shape:
        raise ExtractionError("Target package identity arrays are not aligned")
    if targets.shape != (len(raw_keys), TARGET_POINTS):
        raise ExtractionError(f"Target package spectras has shape {targets.shape}, expected ({len(raw_keys)}, {TARGET_POINTS})")
    expected_energy = np.linspace(ENERGY_MIN_EV, ENERGY_MAX_EV, TARGET_POINTS)
    if energies.shape != (TARGET_POINTS,) or not np.allclose(energies, expected_energy):
        raise ExtractionError("Target package has an unexpected energy grid")
    if split_codes.shape != (len(raw_keys),) or not np.isin(split_codes, [0, 1, 2]).all():
        raise ExtractionError("Target package has invalid split codes")
    keys: list[CuratedKey] = []
    for index, raw_key in enumerate(raw_keys):
        key = parse_curated_key(str(raw_key))
        if str(elements[index]) != key.element or str(materials[index]) != key.material_id or int(sites[index]) != key.site:
            raise ExtractionError(f"Target package identity mismatch at row {index}")
        keys.append(key)
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ExtractionError("Target package contains non-finite or negative targets")
    if len(set(keys)) != len(keys):
        raise ExtractionError("Target package contains duplicate keys")
    return keys, targets, split_codes.astype(np.int8)


def create_target_package(
    spectral_path: Path,
    package_path: Path,
    feature_path: Path | None = None,
    *,
    seed: int = 42,
) -> dict[str, object]:
    """Create a small portable package without reading the Figshare archive."""
    if feature_path is not None:
        source_keys, names = load_aligned_key_archives(feature_path, spectral_path)
    else:
        source_keys, names = load_spectral_key_archive(spectral_path)
    keys, targets, excluded = load_curated_targets(source_keys, names, spectral_path)
    split_by_material, row_counts = assign_material_splits(keys, seed)
    split_codes = np.asarray([split_by_material[key.material_id] for key in keys], dtype=np.int8)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    write_combined_target_archive(package_path, keys, targets, split_codes)
    report = {
        "format_version": 1, "package": str(package_path.resolve()),
        "source_spectra": str(spectral_path.resolve()),
        "source_features": None if feature_path is None else str(feature_path.resolve()),
        "source_row_count": len(source_keys), "excluded_negative_target_count": len(excluded),
        "excluded_negative_targets": excluded, "row_count": len(keys),
        "material_count": len(split_by_material), "target_shape": [len(keys), TARGET_POINTS],
        "target_dtype": "float32", "seed": seed,
        "split_row_counts": dict(zip(SPLIT_NAMES, row_counts)),
    }
    report_path = package_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_manifest(
    path: Path,
    output_dir: Path,
    keys: Sequence[CuratedKey],
    split_codes: np.ndarray,
    candidate_counts: np.ndarray,
    first_source_lines: np.ndarray,
) -> None:
    """Write one row for each curated key in target-array order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
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
            )
        )
        for index, key in enumerate(keys):
            count = int(candidate_counts[index])
            status = "missing" if count == 0 else "matched" if count == 1 else "ambiguous"
            relative_directory = site_directory(output_dir, key).relative_to(output_dir)
            writer.writerow(
                (
                    index,
                    key.text,
                    key.element,
                    key.material_id,
                    key.site,
                    SPLIT_NAMES[int(split_codes[index])],
                    count,
                    "" if first_source_lines[index] < 0 else first_source_lines[index],
                    status,
                    relative_directory.as_posix(),
                )
            )


def write_dataset_readme(output_dir: Path, *, key_only: bool = False) -> None:
    """Document the generated file contract."""
    text = """# Curated AnionXAS Figshare extraction

Source: Figshare article 5678998, file 9932248 (`xas.json.tgz`).

The clean AnionXAS feature and spectrum archives define the accepted
`(element, material_id, site)` keys. Feature values are not copied.

## Layout

```text
extracted/FEFF/<element>/<material_id>/
├── POSCAR                              # only if all matched site structures agree
└── FEFF-XANES/<site>_<element>/
    ├── POSCAR                          # present for one raw candidate
    ├── structure.json                  # present for one raw candidate
    ├── spectrum_raw.dat                # present for one raw candidate
    ├── spectrum_141.dat                # relative energy and curated target
    └── candidates/<number>/...         # all source data when the key is ambiguous
```

`targets_141.npz` contains aligned `keys`, `elements`, `material_ids`, `sites`,
`energies`, `spectras`, and `split_codes` arrays. Split codes 0, 1, and 2 mean
train, validation, and test.

The target grid is relative energy from 0 through 35 eV, inclusive, at 0.25 eV.
The program linearly resamples the clean 200-point AnionXAS target. It does not
change the AnionXAS intensity scale. Every target whose minimum intensity is below
zero is excluded and listed in `report.json` because XASBlock has a nonnegative
Softplus output; negative values are never clipped.

For an ambiguous key, the program moves the first source candidate to
`candidates/000` and writes later candidates beside it. It leaves no unverified
primary structure or raw spectrum in the site directory.

Figshare does not publish the original `feff.inp` or `feff.out` files. This
dataset does not create substitutes for those files. `spectrum_raw.dat` is not
an original `xmu.dat` and must not be passed to the legacy OmniXAS raw FEFF
reader.
"""
    if key_only:
        text += "\n## Key-file mode\n\nThis run used only a CSV keep-list. It intentionally writes no `spectrum_141.dat` or `targets_141.npz`; the raw Figshare structures and spectra are not curated 141-point targets. Provide `--spectra` (or use `--target-package`) when curated targets are required.\n"
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def prepare_output_directory(output_dir: Path, overwrite: bool) -> Path:
    """Create a marked staging directory beside the requested output."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ExtractionError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
        )
    staging_dir = output_dir.with_name(f".{output_dir.name}.partial")
    if staging_dir.exists():
        if not overwrite:
            raise ExtractionError(
                f"Staging directory exists: {staging_dir}. "
                "Inspect it or use --overwrite to replace it."
            )
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    (staging_dir / "INCOMPLETE").write_text(
        "Extraction has not completed successfully.\n", encoding="utf-8"
    )
    return staging_dir


def finalize_output_directory(
    staging_dir: Path, output_dir: Path, overwrite: bool
) -> None:
    """Atomically replace the final directory after successful preparation."""
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite:
            raise ExtractionError(f"Output directory became nonempty: {output_dir}")
        output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except OSError:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def extract_dataset(
    archive_path: Path,
    feature_path: Path | None,
    spectral_path: Path | None,
    output_dir: Path,
    *,
    target_package: Path | None = None,
    seed: int = 42,
    overwrite: bool = False,
    expected_archive_md5: str | None = FIGSHARE_ARCHIVE_MD5,
    allow_incomplete: bool = False,
    max_records: int | None = None,
    key_file: Path | None = None,
) -> dict[str, object]:
    """Build the curated hierarchy and return its report."""
    if not archive_path.is_file():
        raise ExtractionError(f"Required input file does not exist: {archive_path}")
    if key_file is not None and target_package is not None:
        raise ExtractionError("--key-file cannot be combined with --target-package")
    if key_file is not None:
        if not key_file.is_file():
            raise ExtractionError(f"Required input file does not exist: {key_file}")
        if feature_path is not None:
            raise ExtractionError("--key-file mode uses the CSV keep-list; omit --feature-keys")
        source_keys = load_key_file(key_file)
        source_feature_digest = None
        source_spectral_digest = None
        excluded_negative_targets = []
        targets = None
        package_split_codes = None
        if spectral_path is not None:
            if not spectral_path.is_file():
                raise ExtractionError(f"Required input file does not exist: {spectral_path}")
            spectral_keys, names = load_spectral_key_archive(spectral_path)
            names_by_key = {key: names[key] for key in spectral_keys}
            missing = [key.text for key in source_keys if key not in names_by_key]
            if missing:
                raise ExtractionError(f"Key file keys missing from spectra NPZ (first): {missing[:3]}")
            keys, targets, excluded_negative_targets = load_curated_targets(
                source_keys, names_by_key, spectral_path
            )
            source_spectral_digest = file_digest(spectral_path, "sha256")
    elif target_package is not None:
        if not target_package.is_file():
            raise ExtractionError(f"Required input file does not exist: {target_package}")
        source_keys, targets, package_split_codes = load_target_package(target_package)
        excluded_negative_targets: list[dict[str, object]] = []
        source_feature_digest = None
        source_spectral_digest = None
    else:
        if feature_path is None or spectral_path is None:
            raise ExtractionError("--feature-keys and --spectra are required without --target-package")
        for path in (feature_path, spectral_path):
            if not path.is_file():
                raise ExtractionError(f"Required input file does not exist: {path}")
        source_keys, name_by_key = load_aligned_key_archives(feature_path, spectral_path)
        print(f"Validating {len(source_keys):,} curated targets")
        keys, targets, excluded_negative_targets = load_curated_targets(
            source_keys, name_by_key, spectral_path
        )
        source_feature_digest = file_digest(feature_path, "sha256")
        source_spectral_digest = file_digest(spectral_path, "sha256")
    if max_records is not None and max_records < 0:
        raise ExtractionError("max_records must be nonnegative")

    if expected_archive_md5 is not None:
        print(f"Checking MD5 for {archive_path}")
        actual_md5 = file_digest(archive_path, "md5")
        if actual_md5.lower() != expected_archive_md5.lower():
            raise ExtractionError(
                f"Archive MD5 mismatch: expected {expected_archive_md5}, got {actual_md5}"
            )
    else:
        actual_md5 = None

    if key_file is not None:
        keys = source_keys
        split_codes = None
        split_by_material = {key.material_id: None for key in keys}
        split_row_counts = None
    else:
        keys = source_keys if target_package is not None else keys
    working_dir = prepare_output_directory(output_dir, overwrite)

    if target_package is not None:
        split_codes = package_split_codes
        split_by_material = {key.material_id: int(code) for key, code in zip(keys, split_codes)}
        split_row_counts = tuple(int(np.count_nonzero(split_codes == code)) for code in range(3))
    else:
        split_by_material, split_row_counts = assign_material_splits(keys, seed)
        split_codes = np.asarray(
            [split_by_material[key.material_id] for key in keys], dtype=np.int8
        )

    if targets is not None:
        print(f"Writing {len(keys):,} curated 141-point targets")
        write_target_files(working_dir, keys, targets)
        write_combined_target_archive(
            working_dir / "targets_141.npz", keys, targets, split_codes
        )

    candidate_counts = np.zeros(len(keys), dtype=np.uint32)
    first_source_lines = np.full(len(keys), -1, dtype=np.int64)
    print(f"Streaming source records from {archive_path}")
    scan = scan_figshare_archive(
        archive_path,
        working_dir,
        keys,
        candidate_counts,
        first_source_lines,
        max_records=max_records,
    )

    write_manifest(
        working_dir / "manifest.csv",
        working_dir,
        keys,
        split_codes,
        candidate_counts,
        first_source_lines,
    )
    write_dataset_readme(working_dir, key_only=key_file is not None and targets is None)

    element_counts: dict[str, int] = {}
    for key in keys:
        element_counts[key.element] = element_counts.get(key.element, 0) + 1
    missing_indices = np.flatnonzero(candidate_counts == 0)
    ambiguous_indices = np.flatnonzero(candidate_counts > 1)
    report: dict[str, object] = {
        "format_version": 1,
        "figshare_article_id": FIGSHARE_ARTICLE_ID,
        "figshare_file_id": FIGSHARE_FILE_ID,
        "archive": str(archive_path.resolve()),
        "archive_md5": actual_md5,
        "key_file": None if key_file is None else str(key_file.resolve()),
        "key_only": key_file is not None and targets is None,
        "target_package": None if target_package is None else str(target_package.resolve()),
        "curated_feature_keys": None if feature_path is None else str(feature_path.resolve()),
        "curated_spectra": None if spectral_path is None else str(spectral_path.resolve()),
        "curated_feature_sha256": source_feature_digest,
        "curated_spectral_sha256": source_spectral_digest,
        "source_curated_row_count": len(source_keys),
        "excluded_negative_target_count": len(excluded_negative_targets),
        "excluded_negative_targets": excluded_negative_targets,
        "row_count": len(keys),
        "material_count": len(split_by_material),
        "element_count": len(element_counts),
        "element_counts": dict(sorted(element_counts.items())),
        "target_shape": None if targets is None else [len(keys), TARGET_POINTS],
        "target_dtype": None if targets is None else "float32",
        "target_energy_eV": {
            "reference": "relative",
            "start": ENERGY_MIN_EV,
            "end": ENERGY_MAX_EV,
            "spacing": 0.25,
            "points": TARGET_POINTS,
        },
        "target_intensity": "clean AnionXAS scale, unchanged",
        "split": {
            "method": "global material-level greedy row balancing",
            "seed": seed,
            "fractions": dict(zip(SPLIT_NAMES, SPLIT_FRACTIONS)),
            "row_counts": dict(zip(SPLIT_NAMES, split_row_counts)),
        },
        "source_records_read": scan.records_read,
        "matching_raw_candidates": scan.matching_candidates,
        "matched_key_count": int(np.count_nonzero(candidate_counts)),
        "missing_key_count": int(len(missing_indices)),
        "ambiguous_key_count": int(len(ambiguous_indices)),
        "structure_conflict_material_count": len(scan.structure_conflicts),
        "parse_error_count": scan.parse_error_count,
        "record_error_count": scan.record_error_count,
        "missing_key_samples": [keys[index].text for index in missing_indices[:100]],
        "ambiguous_key_samples": [
            {
                "key": keys[index].text,
                "candidate_count": int(candidate_counts[index]),
            }
            for index in ambiguous_indices[:100]
        ],
        "structure_conflict_samples": [
            {"element": element, "material_id": material_id}
            for element, material_id in sorted(scan.structure_conflicts)[:100]
        ],
        "parse_error_samples": scan.parse_error_samples,
        "record_error_samples": scan.record_error_samples,
        "complete": not missing_indices.size
        and scan.parse_error_count == 0
        and scan.record_error_count == 0,
    }
    (working_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not report["complete"] and not allow_incomplete:
        raise ExtractionError(
            "Extraction completed with missing or invalid records. "
            f"See {working_dir / 'report.json'}. Use --allow-incomplete only for inspection."
        )
    if report["complete"]:
        (working_dir / "INCOMPLETE").unlink()
        (working_dir / "COMPLETE").write_text(
            "Extraction completed successfully.\n", encoding="utf-8"
        )
    finalize_output_directory(working_dir, output_dir, overwrite)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream the Figshare K-edge XANES archive and create an "
            "OmniXAS-style hierarchy for clean AnionXAS keys."
        )
    )
    parser.add_argument("--archive", type=Path, help="xas.json.tgz")
    parser.add_argument(
        "--feature-keys", type=Path,
        help="clean_feature_data.npz; values are checked but not copied",
    )
    parser.add_argument("--spectra", type=Path, help="clean_spectral_data.npz")
    parser.add_argument(
        "--key-file", type=Path,
        help="CSV(.gz) keep-list; with no --spectra this writes raw structures/spectra only",
    )
    parser.add_argument(
        "--key-file-output", type=Path,
        help="export key,element,material_id,site CSV from --spectra, excluding negative targets",
    )
    parser.add_argument("--target-package", type=Path, help="portable target NPZ made by --package-only")
    parser.add_argument("--package-only", action="store_true", help="make a portable target package and do not scan Figshare")
    parser.add_argument("--package-output", type=Path, help="output NPZ path for --package-only")
    parser.add_argument("--output", type=Path, help="extraction output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete a nonempty output directory before extraction",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="skip the 5.56 GB archive checksum pass",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return success when selected keys are missing or invalid",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="stop after this many source records; intended only for smoke tests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line extractor."""
    args = build_parser().parse_args(argv)
    try:
        if args.key_file_output is not None:
            if args.spectra is None:
                raise ExtractionError("--key-file-output requires --spectra")
            summary = export_clean_key_file(args.spectra, args.key_file_output)
            print(f"Key file written: {args.key_file_output} ({summary['key_count']:,} keys)")
            print(f"Excluded {summary['excluded_negative_count']:,} negative-target keys")
            return 0
        if args.package_only:
            package_output = args.package_output or args.output
            if args.spectra is None or package_output is None:
                raise ExtractionError("--package-only requires --spectra and --package-output (or --output)")
            report = create_target_package(args.spectra, package_output, args.feature_keys, seed=args.seed)
            print(f"Portable target package written: {package_output}")
            print(f"Report: {package_output.with_suffix('.json')}")
            return 0
        if args.archive is None or args.output is None:
            raise ExtractionError("extraction requires --archive and --output")
        if args.key_file is None and args.target_package is None and (args.feature_keys is None or args.spectra is None):
            raise ExtractionError("extraction requires --feature-keys and --spectra, --target-package, or --key-file")
        if args.key_file is not None and args.target_package is not None:
            raise ExtractionError("--key-file cannot be combined with --target-package")
        report = extract_dataset(
            archive_path=args.archive,
            feature_path=args.feature_keys,
            spectral_path=args.spectra,
            output_dir=args.output,
            target_package=args.target_package,
            seed=args.seed,
            overwrite=args.overwrite,
            expected_archive_md5=None if args.skip_md5 else FIGSHARE_ARCHIVE_MD5,
            allow_incomplete=args.allow_incomplete,
            max_records=args.max_records,
            key_file=args.key_file,
        )
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        "Extraction complete: "
        f"{report['matched_key_count']:,}/{report['row_count']:,} keys matched"
    )
    print(f"Report: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
