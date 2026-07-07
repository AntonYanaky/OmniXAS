#!/usr/bin/env python3
"""Summarize all-8 FEFF pipeline universal_eval.csv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def fnum(x: str) -> float:
    return float(x) if x not in {"", "nan", "None"} else float("nan")


def fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def print_table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(fmt(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", default="output/training/m3gnetAll8FEFF")
    p.add_argument("--out", default=None)
    p.add_argument("--sort-by", default="Cu_test_eta")
    args = p.parse_args()

    base = Path(args.base_dir)
    files = sorted(base.glob("*/universal_eval.csv"))
    if not files:
        raise SystemExit(f"No universal_eval.csv files found under {base}")

    wide_rows, long_rows = [], []
    fieldnames = set()
    for path in files:
        variant = path.parent.name
        with path.open(newline="") as f:
            records = list(csv.DictReader(f))
        if not records:
            continue

        row = {"variant": variant}
        val_etas, test_etas = [], []
        for r in records:
            element = r["element"]
            val_eta = fnum(r["val_eta"])
            test_eta = fnum(r["test_eta"])
            row[f"{element}_val_eta"] = val_eta
            row[f"{element}_test_eta"] = test_eta
            val_etas.append(val_eta)
            test_etas.append(test_eta)
            long_rows.append({"variant": variant, **r})

        row["mean_val_eta"] = mean(val_etas)
        row["mean_test_eta"] = mean(test_etas)
        row["median_val_eta"] = median(val_etas)
        row["median_test_eta"] = median(test_etas)
        fieldnames.update(row)
        wide_rows.append(row)

    sort_by = args.sort_by if args.sort_by in fieldnames else "mean_test_eta"
    wide_rows.sort(key=lambda r: r.get(sort_by, float("-inf")), reverse=True)

    cols = [
        "variant",
        "Cu_val_eta", "Cu_test_eta",
        "mean_val_eta", "mean_test_eta",
        "median_val_eta", "median_test_eta",
    ]
    cols += sorted(c for c in fieldnames if c.endswith("_test_eta") and c not in cols)
    cols = [c for c in cols if c in fieldnames]
    print_table(wide_rows, cols)

    out = Path(args.out) if args.out else base / "all8_feff_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(wide_rows)

    long_out = out.with_name(out.stem + "_long.csv")
    long_fields = sorted(set().union(*(r.keys() for r in long_rows)))
    with long_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(long_rows)

    print(f"\nsaved: {out}")
    print(f"saved: {long_out}")


if __name__ == "__main__":
    main()
