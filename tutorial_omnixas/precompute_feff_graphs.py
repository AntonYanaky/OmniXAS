"""Precompute (graph, line_graph) pairs for the FEFF splits, once per encoder setting.

Builds every material in the train/val/test splits through the exact
``CollateGraphs._build_graphs`` code path (omnixas/data/feff_graph.py), so the
result is identical to the on-the-fly build. One file per task/split, in
canonical ID order: {out_dir}/{task}_{split}.pt = {"header", "ids", "pairs"}.
The header stores cutoff, threebody_cutoff, element_types, matgl/dgl/pymatgen/
torch versions, and a hash of feff_graph.py. ``CachedGraphDataset`` refuses a
file whose header, row counts, or ID order do not match the current setup.

CLI (manual, strict):
    python tutorial_omnixas/precompute_feff_graphs.py [--cutoff 5.0 ...] [--verify]
Library (auto, self-healing; used by the notebooks and the pipeline):
    ensure_graph_cache(...)  # missing -> build, valid -> skip, stale -> loud rebuild
Every encoder setting (cutoff/threebody pair) gets its own directory via
default_out_dir; encoder capacity, seeds, heads, and hyperparameters do not.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from omnixas.data import feff_graph as feff  # noqa: E402


def project_root() -> Path:
    return _HERE.parent


def default_raw_root(root: Path) -> Path:
    import os

    return Path(os.environ.get("OMNIXAS_DATA_ROOT", root.parent / "OmniXAS_data")) / "materialscloud_omnixas_raw" / "extracted"


def default_out_dir(root: Path, cutoff: float, threebody_cutoff: float) -> Path:
    """Cache directory keyed by encoder setting: one set per cutoff/threebody."""
    return root / "output" / "graph_cache" / f"feff_graphs_c{cutoff:g}_tb{threebody_cutoff:g}"


_WORKER: dict = {}


def _worker_init(cutoff: float, threebody_cutoff: float, raw_root: str) -> None:
    feff.patch_matgl_gpu_constants()
    from omnixas.model.m3gnet_xas import M3GNetXASEncoder

    _WORKER["raw"] = Path(raw_root)
    _WORKER["collate"] = feff.CollateGraphs(M3GNetXASEncoder(cutoff=cutoff, threebody_cutoff=threebody_cutoff))


def _worker_job(out_dir: Path, task: str, split: str, chunk: int, ids: list) -> int:
    collate = _WORKER["collate"]
    raw = _WORKER["raw"]
    pairs, out_ids = [], []
    for mid, site in ids:
        path = feff.structure_path(raw, task, mid, int(site))
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw structure for {task} {mid}_{int(site):03d}: {path}")
        structure = feff.Structure.from_file(path) if path.name == "POSCAR" else feff.parse_feff_structure(path)
        pairs.append(collate._build_graphs(structure))
        out_ids.append([mid, int(site)])
    chunk_dir = out_dir / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": out_ids, "pairs": pairs}, chunk_dir / f"{task}_{split}_{chunk:03d}.pt")
    return len(out_ids)


def _element_types() -> list[str]:
    from matgl.config import DEFAULT_ELEMENTS

    return list(DEFAULT_ELEMENTS)


def _build_and_merge(out_dir: Path, task: str, split: str, ids: list, header: dict,
                     cutoff: float, threebody_cutoff: float, raw: Path, workers: int) -> None:
    """Build (task, split) in contiguous chunks across processes, then merge
    the chunks into the single final file in canonical ID order."""
    chunk_dir = out_dir / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    chunk_size = max(1, (n + workers - 1) // workers)
    chunks = [ids[i : i + chunk_size] for i in range(0, n, chunk_size)]
    for i, chunk in enumerate(chunks):
        stale = chunk_dir / f"{task}_{split}_{i:03d}.pt"
        if stale.exists():
            stale.unlink()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(cutoff, threebody_cutoff, str(raw))) as pool:
        futures = {
            pool.submit(_worker_job, out_dir, task, split, i, chunks[i]): (i, len(chunks[i]))
            for i in range(len(chunks))
        }
        for future in as_completed(futures):
            index, size = futures[future]
            done += future.result()
            if done == n:
                print(f"  {task} {split}: all {n} materials built", flush=True)
    if done != n:
        raise RuntimeError(f"{task} {split}: built {done}/{n} materials")
    pairs: list = []
    merged_ids: list = []
    for i in range(len(chunks)):
        payload = torch.load(chunk_dir / f"{task}_{split}_{i:03d}.pt", map_location="cpu", weights_only=False)
        pairs.extend(payload["pairs"])
        merged_ids.extend(payload["ids"])
        payload = None
    if len(pairs) != n or merged_ids != [[m, int(s)] for m, s in ids]:
        raise RuntimeError(f"{task} {split}: chunk merge mismatch (pairs={len(pairs)}, ids={len(merged_ids)})")
    final = out_dir / f"{task}_{split}.pt"
    tmp = out_dir / f".{task}_{split}.pt.tmp"
    torch.save({"header": header, "ids": merged_ids, "pairs": pairs}, tmp)
    pairs = None
    tmp.replace(final)
    for i in range(len(chunks)):
        (chunk_dir / f"{task}_{split}_{i:03d}.pt").unlink()


def verify_cache(root: Path, raw: Path, out_dir: Path, cutoff: float, threebody_cutoff: float,
                 total: int = 20, seed: int = 0) -> int:
    """Guard every cache file, then build ~total sampled materials both ways
    (on-the-fly vs cached) and require exact encoder-feature equality on CPU.
    Raises ValueError on any mismatch. Returns the number of checked materials."""
    if total < 8:
        raise ValueError("total must be at least 8 (one per element)")
    feff.patch_matgl_gpu_constants()
    from omnixas.model.m3gnet_xas import FEATURE_SCALE, M3GNetXASEncoder

    encoder = M3GNetXASEncoder(cutoff=cutoff, threebody_cutoff=threebody_cutoff)
    # full guard: every file, header + row counts + ID order against the canonical files
    datasets = {
        (task, split): feff.CachedGraphDataset(root, out_dir, [task], split, cutoff, threebody_cutoff, encoder.element_types)
        for split in feff.SPLITS
        for task in feff.FEFF_TASKS
    }

    def feats(pair, site):
        g, lg = pair
        return encoder(feff.dgl.batch([g]), feff.dgl.batch([lg]), torch.tensor([int(site)])).mul(FEATURE_SCALE).cpu()

    collate = feff.CollateGraphs(encoder)
    rng = np.random.default_rng(seed)
    checked = 0
    with torch.inference_mode():
        for i, task in enumerate(feff.FEFF_TASKS):
            for j in range(total // len(feff.FEFF_TASKS) + (1 if i < total % len(feff.FEFF_TASKS) else 0)):
                split = feff.SPLITS[j % len(feff.SPLITS)]
                ds = datasets[(task, split)]
                idx = int(rng.integers(len(ds)))
                _, mid, site, _ = ds.rows[idx]
                path = feff.structure_path(raw, task, mid, int(site))
                if not path.is_file():
                    raise FileNotFoundError(f"Missing raw structure: {path}")
                structure = feff.Structure.from_file(path) if path.name == "POSCAR" else feff.parse_feff_structure(path)
                cached = feats(ds.pairs[idx], site)
                live = feats(collate._build_graphs(structure), site)
                if not torch.equal(cached, live):
                    raise ValueError(
                        f"Graph cache verification failed: {task} {split} {mid}_{int(site):03d} "
                        "features differ between cached and on-the-fly builds. Rerun precompute."
                    )
                checked += 1
    print(f"VERIFIED: {checked} sampled materials match the on-the-fly build exactly (encoder features, CPU).", flush=True)
    return checked


def ensure_graph_cache(
    root: Path,
    raw: Path,
    out_dir: Path,
    cutoff: float,
    threebody_cutoff: float,
    workers: int = 32,
    overwrite: bool = False,
    rebuild_stale: bool = False,
    tasks: list[str] | None = None,
    splits: list[str] | None = None,
    verify: bool = True,
) -> None:
    """Build or refresh the precomputed graph cache for one encoder setting.

    Missing files are built; valid files are skipped; stale files (header or
    ID mismatch) raise unless rebuild_stale, in which case they are deleted
    and rebuilt with a loud log (never silently reused). overwrite rebuilds
    every file. verify runs verify_cache afterwards. Consumers re-check every
    file at load time, so this is a cache-builder, not the source of truth."""
    root, raw, out_dir = Path(root), Path(raw), Path(out_dir)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    tasks = list(tasks) if tasks else list(feff.FEFF_TASKS)
    splits = list(splits) if splits else list(feff.SPLITS)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = feff.build_graph_header(cutoff, threebody_cutoff, _element_types())
    print(f"graph cache: cutoff={cutoff} threebody_cutoff={threebody_cutoff} out={out_dir}")

    start = time.time()
    total_mb = 0.0
    for task in tasks:
        for split in splits:
            ids = feff.load_id_rows(root, task, split)
            final = out_dir / f"{task}_{split}.pt"
            rebuild = overwrite
            if final.is_file() and not overwrite:
                payload = torch.load(final, map_location="cpu", weights_only=False)
                problems = feff.graph_header_problems(payload.get("header", {}), header)
                cache_ids = [(str(m), int(s)) for m, s in payload.get("ids", [])]
                if len(cache_ids) != len(ids):
                    problems.append(f"row counts differ: id_file={len(ids)} cache_ids={len(cache_ids)}")
                elif cache_ids != ids:
                    problems.append("material/site IDs or their order do not match the canonical split ID file")
                if problems:
                    if not rebuild_stale:
                        raise ValueError(
                            f"Stale graph cache file {final}:\n  " + "\n  ".join(problems)
                            + "\nRerun with --overwrite (or rebuild_stale=True) to rebuild."
                        )
                    print(f"STALE graph cache {final}:\n  " + "\n  ".join(problems) + "\n  -> rebuilding (loud, not silent)", flush=True)
                    final.unlink()
                    rebuild = True
            if final.is_file() and not rebuild:
                total_mb += final.stat().st_size / 1e6
                print(f"skip {task} {split}: {len(ids)} rows, {final.stat().st_size / 1e6:.1f} MB (guard passed)", flush=True)
                continue
            if not raw.is_dir():
                raise FileNotFoundError(f"Missing raw FEFF structure root: {raw}. Set OMNIXAS_DATA_ROOT. Needed to build {final}.")
            _build_and_merge(out_dir, task, split, ids, header, cutoff, threebody_cutoff, raw, workers)
            total_mb += final.stat().st_size / 1e6
            print(f"built {task} {split}: {len(ids)} rows, {total_mb / 1e3:.1f} GB total so far ({time.time() - start:.0f}s elapsed)", flush=True)

    chunk_dir = out_dir / "_chunks"
    if chunk_dir.is_dir():
        leftovers = list(chunk_dir.iterdir())
        if leftovers:
            raise RuntimeError(f"Leftover chunk files: {leftovers[:4]}")
        chunk_dir.rmdir()
    print(f"graph cache ready in {time.time() - start:.0f}s. Total: {total_mb / 1e3:.1f} GB "
          "(training with this cache holds the train+val files in main-process RAM).")
    if verify:
        verify_cache(root, raw, out_dir, cutoff, threebody_cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="OmniXAS repo root (default: parent of this script)")
    parser.add_argument("--raw-root", default=None, help="Raw structure root (default: $OMNIXAS_DATA_ROOT/... or ../OmniXAS_data/...)")
    parser.add_argument("--out-dir", default=None, help="Cache directory (default: output/graph_cache/feff_graphs_c{cutoff}_tb{threebody})")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threebody-cutoff", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild every file, even ones that pass the guard")
    parser.add_argument("--verify", action="store_true", help="Run the exact-build verification after the run")
    parser.add_argument("--tasks", nargs="*", default=list(feff.FEFF_TASKS))
    parser.add_argument("--splits", nargs="*", default=list(feff.SPLITS))
    args = parser.parse_args()
    root = Path(args.root) if args.root else project_root()
    raw = Path(args.raw_root) if args.raw_root else default_raw_root(root)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(root, args.cutoff, args.threebody_cutoff)
    ensure_graph_cache(
        root, raw, out_dir, args.cutoff, args.threebody_cutoff,
        workers=args.workers, overwrite=args.overwrite, rebuild_stale=False,
        tasks=args.tasks, splits=args.splits, verify=args.verify,
    )
    if not args.verify:
        print("Verify before use: rerun with --verify (the notebooks and the pipeline verify automatically).")


if __name__ == "__main__":
    main()
