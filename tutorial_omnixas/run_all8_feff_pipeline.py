#!/usr/bin/env python3
"""Run the all-8 FEFF encoder -> UniversalXAS -> evaluation pipeline.

Examples:
  python tutorial_omnixas/run_all8_feff_pipeline.py --variants pretrained_temp
  python tutorial_omnixas/run_all8_feff_pipeline.py --variants pretrained_temp pretrained_frozen_universal scratch_temp
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VARIANTS = {
    "pretrained_temp": {
        "run_name": "pretrained_temp_b2_seed42",
        "encoder": [
            "--init", "pretrained",
            "--supervisor", "temp_head",
            "--epochs", "100",
            "--patience", "20",
            "--train-blocks", "2",
            "--encoder-lr", "2e-6",
            "--head-lr", "1e-3",
            "--anchor-lambda", "1e-4",
        ],
    },
    "pretrained_frozen_universal": {
        "run_name": "pretrained_frozen_universal_b2_seed42",
        "needs_source_universal": True,
        "encoder": [
            "--init", "pretrained",
            "--supervisor", "frozen_universal",
            "--epochs", "80",
            "--patience", "20",
            "--train-blocks", "2",
            "--encoder-lr", "2e-6",
            "--anchor-lambda", "1e-4",
        ],
    },
    "scratch_temp": {
        "run_name": "scratch_temp_seed42",
        "encoder": [
            "--init", "scratch",
            "--supervisor", "temp_head",
            "--epochs", "300",
            "--patience", "30",
            "--encoder-lr", "1e-3",
            "--head-lr", "1e-3",
            "--anchor-lambda", "0.0",
        ],
    },
}

UNIVERSAL_ARGS = [
    "--models", "universal",
    "--seed", "42",
    "--universal-lr", "7e-4",
    "--universal-dropout", "0.25",
    "--universal-max-epochs", "800",
    "--universal-patience", "60",
    "--universal-monitor", "val_median_mse",
    "--universal-cos-lr",
    "--universal-shuffle",
    "--no-progress-bar",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["pretrained_temp"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", default="24")
    parser.add_argument("--num-workers", default="4")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--deriv-lambda", default="0.02")
    parser.add_argument("--base-dir", default="output/training/m3gnetAll8FEFF")
    parser.add_argument("--source-universal-ckpt", default=None, help="Checkpoint for frozen-Universal supervisor. Defaults to latest existing UniversalXAS.")
    parser.add_argument("--train-source-universal", action="store_true", help="Train a new source UniversalXAS instead of reusing an existing one.")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose expected outputs already exist.")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to test the pipeline wiring.")
    return parser.parse_args()


def run(cmd: list[str], *, log_path: Path, env: dict[str, str], cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"Command failed with exit code {code}. See {log_path}")


def require_paths(root: Path) -> None:
    data_root = os.environ.get("OMNIXAS_DATA_ROOT")
    if not data_root:
        raise SystemExit('Set OMNIXAS_DATA_ROOT first, e.g. echo \'export OMNIXAS_DATA_ROOT="$HOME/OmniXAS_data"\' >> ~/.bashrc')
    raw_feff = Path(data_root) / "materialscloud_omnixas_raw" / "extracted" / "FEFF"
    if not raw_feff.exists():
        raise SystemExit(f"Missing raw FEFF data: {raw_feff}\nRun: bash tutorial_omnixas/download_omnixas_raw_data.sh")
    if not (root / "models" / "M3GNet-MP-2021.2.8-PES").exists():
        raise SystemExit("Missing models/M3GNet-MP-2021.2.8-PES")


def latest_universal_ckpt(root: Path, training_root: Path) -> Path | None:
    ckpts = []
    for base in [training_root, root / "output" / "training"]:
        ckpts += list((base / "universalXAS" / "All_FEFF" / "runs").glob("*/best*.ckpt"))
        ckpts += list((base / "universalXAS" / "All_FEFF" / "checkpoints").glob("best*.ckpt"))
    return sorted(set(ckpts), key=lambda p: p.stat().st_mtime)[-1] if ckpts else None


def source_universal_ckpt(args: argparse.Namespace, root: Path, env: dict[str, str], base_dir: Path) -> Path:
    if args.source_universal_ckpt:
        ckpt = Path(args.source_universal_ckpt)
        if not ckpt.exists():
            raise SystemExit(f"Missing --source-universal-ckpt: {ckpt}")
        return ckpt

    source_root = base_dir / "source_universal"
    if not args.train_source_universal:
        ckpt = latest_universal_ckpt(root, source_root)
        if ckpt is not None:
            return ckpt
        raise SystemExit("No existing UniversalXAS checkpoint found. Run with --train-source-universal or pass --source-universal-ckpt.")

    run(
        [
            sys.executable,
            "tutorial_omnixas/train_paper_models.py",
            *UNIVERSAL_ARGS,
            "--data-dir", "tutorial_omnixas/ml_data",
            "--training-root", str(source_root),
        ],
        log_path=source_root / "source_universal.log",
        env=env,
        cwd=root,
    )
    ckpt = latest_universal_ckpt(root, source_root)
    if ckpt is None:
        raise SystemExit(f"No UniversalXAS checkpoint created under {source_root}")
    return ckpt


def variant_encoder_args(name: str, args: argparse.Namespace, source_ckpt: Path | None) -> tuple[str, list[str]]:
    config = VARIANTS[name]
    run_name = "smoke_" + config["run_name"] if args.smoke else config["run_name"]
    enc = list(config["encoder"])
    if args.smoke:
        cleaned = []
        skip_next = False
        for item in enc:
            if skip_next:
                skip_next = False
                continue
            if item in {"--epochs", "--patience"}:
                skip_next = True
                continue
            cleaned.append(item)
        enc = cleaned + ["--epochs", "2", "--patience", "1", "--max-train-per-task", "64", "--max-val-per-task", "32"]
    if config.get("needs_source_universal"):
        if source_ckpt is None:
            raise SystemExit(f"Variant {name} needs a source UniversalXAS checkpoint")
        enc += ["--universal-ckpt", str(source_ckpt)]
    enc += [
        "--batch-size", args.batch_size,
        "--num-workers", "0" if args.smoke else args.num_workers,
        "--accelerator", args.accelerator,
        "--deriv-lambda", args.deriv_lambda,
        "--run-name", run_name,
    ]
    return run_name, enc


def run_variant(name: str, args: argparse.Namespace, root: Path, env: dict[str, str], base_dir: Path, source_ckpt: Path | None) -> None:
    run_name, enc_args = variant_encoder_args(name, args, source_ckpt)
    run_dir = base_dir / run_name
    features = run_dir / "features"
    eval_csv = run_dir / "universal_eval.csv"

    print(f"\n=== {name}: encoder ===", flush=True)
    if not (args.resume and features.exists()):
        run(
            [sys.executable, "tutorial_omnixas/m3gnet_all8_feff_encoder.py", *enc_args],
            log_path=run_dir / "encoder.log",
            env=env,
            cwd=root,
        )

    print(f"\n=== {name}: UniversalXAS ===", flush=True)
    heads = run_dir / "heads"
    if not (args.resume and list((heads / "universalXAS" / "All_FEFF" / "runs").glob("*/best*.ckpt"))):
        run(
            [
                sys.executable,
                "tutorial_omnixas/train_paper_models.py",
                *UNIVERSAL_ARGS,
                "--data-dir", str(features),
                "--training-root", str(heads),
            ],
            log_path=run_dir / "universal.log",
            env=env,
            cwd=root,
        )

    print(f"\n=== {name}: eval ===", flush=True)
    if not (args.resume and eval_csv.exists()):
        run(
            [
                sys.executable,
                "tutorial_omnixas/evaluate_universal_features.py",
                "--data-dir", str(features),
                "--run-root", str(heads),
                "--out", str(eval_csv),
            ],
            log_path=run_dir / "eval.log",
            env=env,
            cwd=root,
        )
    print(eval_csv.read_text(), flush=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    base_dir = root / args.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    require_paths(root)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TQDM_DISABLE"] = "1"

    print("plan:", " -> ".join(args.variants), flush=True)
    if any(VARIANTS[name].get("needs_source_universal") for name in args.variants):
        print("note: pretrained_frozen_universal reuses an existing UniversalXAS checkpoint unless --train-source-universal is set.", flush=True)

    source_ckpt = None
    if any(VARIANTS[name].get("needs_source_universal") for name in args.variants):
        print("=== source UniversalXAS ===", flush=True)
        source_ckpt = source_universal_ckpt(args, root, env, base_dir)
        print(f"source checkpoint: {source_ckpt}", flush=True)

    print("started:", datetime.now().isoformat(timespec="seconds"), flush=True)
    for name in args.variants:
        run_variant(name, args, root, env, base_dir, source_ckpt)
    print("finished:", datetime.now().isoformat(timespec="seconds"), flush=True)


if __name__ == "__main__":
    main()
