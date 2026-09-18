# Build a curated AnionXAS raw-data hierarchy

**Scientific default:** train new scratch models on the native clean 200-point targets. Use 141 points only as an explicitly resampled comparison.

This workflow selects AnionXAS absorber-site keys and matching structures from Figshare article 5678998. The archive contains processed records, not original FEFF `feff.inp` or `feff.out` files.

## Compact key-file workflow (recommended for committed keep-lists)

Build the committed keep-list locally: `python tutorial_omnixas/extract_figshare_anionxas.py --key-file-output tutorial_omnixas/anionxas_clean_keys.csv.gz --spectra C:/Users/anton/Desktop/drive-download-20260804T124325Z-1-001/clean_spectral_data.npz`

Export a small CSV from the clean spectra. Every NPZ member whose 200-point target contains a negative value is excluded:

```bash
python tutorial_omnixas/extract_figshare_anionxas.py \
  --spectra /path/to/clean_spectral_data.npz \
  --key-file-output tutorial_omnixas/anionxas_clean_keys.csv.gz
```

Scan Figshare using only that keep-list (key-only extraction): `python tutorial_omnixas/extract_figshare_anionxas.py --archive /path/to/xas.json.tgz --key-file tutorial_omnixas/anionxas_clean_keys.csv.gz --output /path/to/anionxas_raw --skip-md5`

This writes raw source structures/spectra, `manifest.csv`, and a report, but **does not write curated `spectrum_141.dat` or `targets_141.npz`** because a key file has no target arrays:

```bash
python tutorial_omnixas/extract_figshare_anionxas.py \
  --archive /path/to/xas.json.tgz \
  --key-file tutorial_omnixas/anionxas_clean_keys.csv.gz \
  --output /path/to/anionxas_raw \
  --skip-md5
```

When curated 141-point targets are required, provide the spectra NPZ alongside the key file (or use the package workflow below). Target-package extraction: `python tutorial_omnixas/extract_figshare_anionxas.py --archive /path/to/xas.json.tgz --target-package tutorial_omnixas/anionxas_targets_141.npz --output /path/to/anionxas_curated_141 --skip-md5`

To rerun an existing extraction after this patch, add `--overwrite`: `python tutorial_omnixas/extract_figshare_anionxas.py --archive /path/to/xas.json.tgz --key-file tutorial_omnixas/anionxas_clean_keys.csv.gz --output /path/to/anionxas_raw --skip-md5 --overwrite`

```bash
python tutorial_omnixas/extract_figshare_anionxas.py \
  --archive /path/to/xas.json.tgz \
  --key-file tutorial_omnixas/anionxas_clean_keys.csv.gz \
  --spectra /path/to/clean_spectral_data.npz \
  --output /path/to/anionxas_curated_141
```

The CSV columns are exactly `key,element,material_id,site`. It is a Figshare keep-list, not a replacement for the clean spectra NPZ.

## Portable target-package workflow

```bash
python tutorial_omnixas/extract_figshare_anionxas.py \
  --package-only \
  --spectra C:/Users/anton/Desktop/drive-download-20260804T124325Z-1-001/clean_spectral_data.npz \
  --feature-keys C:/Users/anton/Desktop/drive-download-20260804T124325Z-1-001/clean_feature_data.npz \
  --target-points 200 \
  --package-output tutorial_omnixas/anionxas_targets_200.npz
```

Native 200-point package (preferred for new scratch training):
`python tutorial_omnixas/extract_figshare_anionxas.py --package-only --spectra C:/Users/anton/Desktop/drive-download-20260804T124325Z-1-001/clean_spectral_data.npz --target-points 200 --package-output tutorial_omnixas/anionxas_targets_200.npz`

Resampled 141-point comparison package (preserves the existing workflow):
`python tutorial_omnixas/extract_figshare_anionxas.py --package-only --spectra C:/Users/anton/Desktop/drive-download-20260804T124325Z-1-001/clean_spectral_data.npz --target-points 141 --package-output tutorial_omnixas/anionxas_targets_141.npz`

Transfer the package and its adjacent JSON report to the extraction machine, together with `xas.json.tgz`:

```bash
python tutorial_omnixas/extract_figshare_anionxas.py \
  --archive /scratch/$USER/xas.json.tgz \
  --target-package /scratch/$USER/anionxas_targets_141.npz \
  --output /scratch/$USER/anionxas_curated_141
```

The archive MD5 is checked by default. Use `--skip-md5` only after an independent trusted check.

## Direct workflow (without a package)

```bash
python tutorial_omnixas/extract_figshare_anionxas.py \
  --archive /path/to/xas.json.tgz \
  --feature-keys /path/to/clean_feature_data.npz \
  --spectra /path/to/clean_spectral_data.npz \
  --output /path/to/anionxas_curated_141
```

The clean feature and spectrum archives must have exactly the same keys. Twenty clean targets contain negative intensity and are excluded; negative values are never clipped.

## Outputs and target definition

The extraction output contains `report.json`, `manifest.csv`, and an `extracted/FEFF/<element>/<material_id>/FEFF-XANES/<site>_<element>/` hierarchy with source structures and raw spectra. Target-enabled runs contain `targets_200.npz`/`spectrum_200.dat` for native clean 200 points, or `targets_141.npz`/`spectrum_141.dat` for linearly resampled 141 points at 0.25 eV. Reports and package metadata identify `native clean 200` versus `resampled 141`. **Warning:** train a 200-output model for native curation; train a 141-output model only as a resampled comparison.

The extractor reports missing keys, ambiguous source records, and structure conflicts. It refuses to replace a nonempty output directory unless `--overwrite` is set; use `--allow-incomplete` only for inspection.
