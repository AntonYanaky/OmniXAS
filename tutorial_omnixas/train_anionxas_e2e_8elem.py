#!/usr/bin/env python3
"""End-to-end eight-element AnionXAS training pipeline.

The default is a full scratch M3GNetXAS encoder run, followed by a UniversalXAS
head and validation-selected per-element tuned heads.  ``--preflight`` performs
only lightweight package/path checks and therefore does not import MatGL.
"""
from __future__ import annotations
import argparse, json, os, random, shutil
from pathlib import Path
import numpy as np

ELEMENTS = ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu")
SPLITS = ("train", "val", "test")
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
TARGET_DIM = 200
FEATURE_DIM = 64
STRUCTURE_POLICY = "site POSCAR when present; otherwise candidates/000/POSCAR; ambiguous candidates use 000"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-package", type=Path, default=Path("tutorial_omnixas/anionxas_targets_200.npz"))
    p.add_argument("--data-root", type=Path, default=None, help="defaults to $OMNIXAS_DATA_ROOT")
    p.add_argument("--output-root", type=Path, default=Path("output/anionxas_e2e_8elem"))
    p.add_argument("--run-name", default="seed42")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--gpu", default=None)
    p.add_argument("--num-workers", type=int, default=0); p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--encoder-epochs", type=int, default=None); p.add_argument("--head-epochs", type=int, default=800)
    p.add_argument("--tuned-epochs", type=int, default=1000); p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--encoder-batch-size", type=int, default=24); p.add_argument("--head-patience", type=int, default=60)
    p.add_argument("--preflight", action="store_true"); p.add_argument("--evaluate", action="store_true")
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def project_root(): return Path(__file__).resolve().parents[1]
def raw_root(args):
    root = args.data_root or Path(os.environ.get("OMNIXAS_DATA_ROOT", project_root().parent / "OmniXAS_data"))
    return root / "anionxas_curated_200" / "extracted" / "FEFF"


def structure_candidates(raw, element, material, site):
    # Extraction versions differ in whether POSCAR is stored at site or material level.
    material_dir = raw / element / material
    site_dir = material_dir / "FEFF-XANES" / f"{site:03d}_{element}"
    return (site_dir / "POSCAR", site_dir / "candidates" / "000" / "POSCAR",
            material_dir / "POSCAR", material_dir / "candidates" / "000" / "POSCAR")


def load_rows(package, raw):
    """Validate the native package, then return rows with available structures.

    The package may contain elements outside this eight-element workflow; those
    rows are intentionally ignored after validating the package-wide invariants.
    """
    with np.load(package, allow_pickle=False) as z:
        required = {"keys", "elements", "material_ids", "sites", "energies",
                    "spectras", "split_codes", "target_points", "target_provenance"}
        missing_keys = required - set(z.files)
        if missing_keys:
            raise ValueError(f"target package missing {sorted(missing_keys)}")
        keys = np.asarray(z["keys"])
        elements, materials, sites = z["elements"], z["material_ids"], z["sites"]
        energies = np.asarray(z["energies"], dtype=np.float64)
        y = np.asarray(z["spectras"], dtype=np.float32)
        codes = np.asarray(z["split_codes"])
        target_points = np.asarray(z["target_points"])
        provenance = np.asarray(z["target_provenance"])
    n = len(y)
    if keys.ndim != 1 or len(keys) != n: raise ValueError("keys must be a row-aligned 1D array")
    if y.ndim != 2 or y.shape != (n, TARGET_DIM): raise ValueError(f"expected targets (n,200), got {y.shape}")
    if not (len(elements) == len(materials) == len(sites) == len(codes) == n):
        raise ValueError("package arrays are not row aligned")
    expected_grid = np.linspace(0.0, 35.0, TARGET_DIM)
    if energies.shape != (TARGET_DIM,) or not np.all(np.isfinite(energies)) or not np.allclose(energies, expected_grid, rtol=0, atol=1e-5):
        raise ValueError("energies must be the finite 200-point linspace grid from 0 to 35")
    if target_points.shape != () or int(target_points) != TARGET_DIM:
        raise ValueError(f"target_points must be scalar {TARGET_DIM}")
    if provenance.shape != () or not str(provenance).strip():
        raise ValueError("target_provenance must be a non-empty scalar")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("spectras must contain finite, nonnegative targets")
    if not np.issubdtype(sites.dtype, np.integer) or np.any(sites < 0):
        raise ValueError("sites must be integer and nonnegative")
    if not np.issubdtype(codes.dtype, np.integer) or np.any(~np.isin(codes, tuple(SPLIT_NAMES))):
        raise ValueError("split_codes must be integer values 0, 1, or 2")
    material_splits = {}
    for i, (material, code) in enumerate(zip(materials, codes, strict=True)):
        material = str(material)
        if not material: raise ValueError(f"empty material_id at row {i}")
        previous = material_splits.setdefault(material, int(code))
        if previous != int(code):
            raise ValueError(f"material {material!r} has multiple split codes")

    rows, missing = [], []
    for i, (element, material, site, target, code) in enumerate(zip(elements, materials, sites, y, codes, strict=True)):
        element, material, site = str(element), str(material), int(site)
        if element not in ELEMENTS: continue
        paths = structure_candidates(raw, element, material, site)
        found = next((x for x in paths if x.is_file()), None)
        if found is None:
            missing.append({"index": i, "element": element, "material_id": material, "site": site, "candidates": [str(x) for x in paths]})
            continue
        rows.append((element, material, site, int(code), target, found))
    return rows, missing


def preflight(args):
    package = args.target_package if args.target_package.is_absolute() else project_root()/args.target_package
    raw = raw_root(args)
    if not package.is_file(): raise FileNotFoundError(package)
    if not raw.is_dir(): raise FileNotFoundError(f"Missing AnionXAS structure root: {raw}")
    rows, missing = load_rows(package, raw)
    counts = {e: {s: sum(r[0] == e and SPLIT_NAMES[r[3]] == s for r in rows) for s in SPLITS} for e in ELEMENTS}
    empty = [(e, s) for e in ELEMENTS for s in SPLITS if counts[e][s] == 0]
    if not rows: raise ValueError("no usable eight-element rows remain after missing-structure filtering")
    if empty:
        raise ValueError("empty element/split counts after missing-structure filtering: " + ", ".join(f"{e}/{s}" for e, s in empty))
    print(json.dumps({"package": str(package), "raw_root": str(raw), "usable_rows": len(rows), "skipped_missing_structure": len(missing), "counts": counts, "structure_policy": STRUCTURE_POLICY, "target_dim": TARGET_DIM, "split_source": "package split_codes (global material split)"}, indent=2))
    if missing: print("Skipped rows (first 10):\n" + "\n".join(json.dumps(x) for x in missing[:10]))
    return package, raw, rows, missing


def main():
    args = parse_args()
    if args.resume:
        raise NotImplementedError("--resume is not implemented; use --overwrite or choose a new --run-name")
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    package, raw, rows, missing = preflight(args)
    if args.preflight: return
    root = args.output_root if args.output_root.is_absolute() else project_root()/args.output_root
    run = root / args.run_name
    if args.overwrite and run.exists(): shutil.rmtree(run)
    if args.evaluate:
        metrics = run / "metrics.csv"
        if not metrics.is_file() or not (run / "RUN_COMPLETE.json").is_file():
            raise RuntimeError(f"Completed run artifacts are missing under {run}")
        print(metrics.read_text(encoding="utf-8"), end="")
        return
    if run.exists(): raise FileExistsError(f"{run} exists; use --overwrite or choose a new --run-name")
    run.mkdir(parents=True, exist_ok=True)
    (run/"provenance.json").write_text(json.dumps({"pipeline":"AnionXAS_e2e_8elem", "elements":ELEMENTS, "target_package":str(package), "raw_root":str(raw), "target_dim":200, "native_grid":"200 points", "split_source":"split_codes from target package", "selection":"validation only; test evaluated once", "structure_policy":STRUCTURE_POLICY, "skipped_missing_structure":missing, "args":vars(args)}, indent=2, default=str), encoding="utf-8")
    _train(args, run, rows)


class GraphRows:
    """Pickle-safe dataset used by DataLoader workers."""
    def __init__(self, rows, split):
        self.rows = [r for r in rows if SPLIT_NAMES[r[3]] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        e, m, site, code, y, path = self.rows[i]
        from pymatgen.core import Structure
        import torch
        return f"{e}_FEFF", m, Structure.from_file(path), site, torch.tensor(y)


def _train(args, run, rows):
    # Heavy dependencies are deliberately imported here, keeping --preflight/import safe.
    os.environ.setdefault("DGLBACKEND", "pytorch")
    import torch
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from omnixas.data.feff_graph import CollateGraphs, patch_matgl_gpu_constants
    from omnixas.model.m3gnet_xas import M3GNetXASEncoder, XASSpectralHead, FEATURE_SCALE, HEAD_HIDDEN_DIMS
    patch_matgl_gpu_constants()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); pl.seed_everything(args.seed, workers=True)

    # A 200-output head is used for the end-to-end encoder (not the published 141 head).
    class E2E(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder=M3GNetXASEncoder(); self.head=XASSpectralHead(output_dim=TARGET_DIM); self.feature_scale=FEATURE_SCALE
        def encode(self,g,l,s): return self.encoder(g,l,s)*self.feature_scale
        def forward(self,g,l,s): return self.head(self.encode(g,l,s))
    encoder = E2E(); collate = CollateGraphs(encoder.encoder)
    loaders = {s: DataLoader(GraphRows(rows, s), batch_size=args.encoder_batch_size if s != "train" else args.batch_size, shuffle=s=="train", num_workers=args.num_workers, collate_fn=collate) for s in SPLITS}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); encoder.to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3); best=float("inf"); best_path=run/"best_encoder.pt"
    epochs = args.encoder_epochs or args.epochs
    progress = tqdm(range(epochs), desc="Encoder", unit="epoch")
    for epoch in progress:
        encoder.train()
        for b in loaders["train"]:
            opt.zero_grad(); pred=encoder(b["graph"].to(device),b["line_graph"].to(device),b["site"].to(device)); loss=(pred-b["y"].to(device)).square().mean(); loss.backward(); opt.step()
        encoder.eval(); vals=[]
        with torch.inference_mode():
            for b in loaders["val"]: vals.append((encoder(b["graph"].to(device),b["line_graph"].to(device),b["site"].to(device))-b["y"].to(device)).square().mean().item())
        val=float(np.mean(vals)) if vals else float("inf")
        if val < best: best=val; torch.save({"state_dict":encoder.state_dict(),"epoch":epoch,"val_loss":val},best_path)
        progress.set_postfix(val_mse=f"{val:.2e}", best=f"{best:.2e}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        if epoch % 10 == 0: tqdm.write(f"encoder epoch {epoch}: val_mse={val:.5g}", flush=True)
    progress.close()
    encoder.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)["state_dict"]); encoder.eval()
    features={s: [] for s in SPLITS}; targets={s: [] for s in SPLITS}; elems={s: [] for s in SPLITS}
    with torch.inference_mode():
        for s in tqdm(SPLITS, desc="Feature export", unit="split"):
            for b in DataLoader(GraphRows(rows, s), batch_size=args.encoder_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate):
                features[s].append(encoder.encode(b["graph"].to(device),b["line_graph"].to(device),b["site"].to(device)).cpu().numpy()); targets[s].append(b["y"].numpy())
    fdir=run/"features"; fdir.mkdir(exist_ok=True)
    for s in SPLITS:
        features[s]=np.concatenate(features[s]); targets[s]=np.concatenate(targets[s]); np.savez(fdir/f"{s}.npz", X=features[s], y=targets[s])
    # Element labels are retained from row order (the graph loader preserves it).
    for s in SPLITS: elems[s]=np.array([r[0] for r in rows if SPLIT_NAMES[r[3]]==s])
    def fit_head(name, X,y,V,Y, source=None, epochs=args.head_epochs):
        head=XASSpectralHead(output_dim=TARGET_DIM).to(device)
        if source: head.load_state_dict(source)
        o=torch.optim.Adam(head.parameters(),lr=5e-4); bestv=float("inf"); path=run/"heads"/name/"best.pt"; path.parent.mkdir(parents=True,exist_ok=True)
        progress = tqdm(range(epochs), desc=name, unit="epoch")
        for ep in progress:
            head.train()
            for ix in torch.randperm(len(X)) .split(args.batch_size):
                o.zero_grad(); z=head(torch.as_tensor(X[ix],device=device)); (z-torch.as_tensor(y[ix],device=device)).square().mean().backward(); o.step()
            head.eval()
            with torch.inference_mode():
                v=(head(torch.as_tensor(V,device=device))-torch.as_tensor(Y,device=device)).square().mean().item()
            if v<bestv: bestv=v; torch.save({"state_dict":head.state_dict(),"val_mse":v,"epoch":ep},path)
            progress.set_postfix(val=f"{v:.2e}", best=f"{bestv:.2e}", lr=f"{o.param_groups[0]['lr']:.2e}")
        progress.close()
        tqdm.write(f"[{name}] done, checkpoint: {path}")
        return path
    allX,ally=np.concatenate([features[s] for s in ("train",)]),targets["train"]
    universal=fit_head("universalXAS",allX,ally,np.concatenate([features["val"]]),np.concatenate([targets["val"]]))
    state=torch.load(universal,map_location="cpu",weights_only=False)["state_dict"]
    for e in ELEMENTS:
        tr=np.where(elems["train"]==e)[0]; va=np.where(elems["val"]==e)[0]
        if len(tr) and len(va): fit_head(f"tunedUniversalXAS/{e}",features["train"][tr],targets["train"][tr],features["val"][va],targets["val"][va],state,args.tuned_epochs)
    _evaluate(run, features, targets, elems, args.batch_size, device, universal)
    (run/"RUN_COMPLETE.json").write_text(json.dumps({"status":"complete","selection":"validation only; test evaluated once"},indent=2),encoding="utf-8")


def _evaluate(run, features, targets, elems, batch, device, universal):
    import csv, torch
    from omnixas.model.m3gnet_xas import XASSpectralHead
    rows=[]
    for variant, paths in [("UniversalXAS", {e:universal for e in ELEMENTS}), ("Tuned-UniversalXAS", {e:run/"heads"/"tunedUniversalXAS"/e/"best.pt" for e in ELEMENTS})]:
        for e,path in paths.items():
            if not path.is_file(): continue
            h=XASSpectralHead(output_dim=TARGET_DIM).to(device); h.load_state_dict(torch.load(path,map_location=device,weights_only=False)["state_dict"]); h.eval()
            for split in ("val","test"):
                ix=np.where(elems[split]==e)[0]
                if not len(ix): continue
                with torch.inference_mode(): pred=h(torch.as_tensor(features[split][ix],device=device)).cpu().numpy()
                mse=np.mean((pred-targets[split][ix])**2,axis=1); baseline=np.median(np.mean((targets[split][ix]-targets["train"][elems["train"]==e].mean(0))**2,axis=1))
                rows.append({"element":e,"variant":variant,"split":split,"mse":float(mse.mean()),"median_mse":float(np.median(mse)),"eta":float(baseline/max(np.median(mse),1e-12))})
    with (run/"metrics.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__ == "__main__": main()
