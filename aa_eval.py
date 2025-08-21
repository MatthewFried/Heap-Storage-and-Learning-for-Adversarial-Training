#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

# --- import attack helpers from your training code ---
from learning_heap import Attacks, get_cifar10_loaders, Config

# -------------------- utils --------------------
def resolve_latest_dir(prefix_or_path: str, root: Path) -> Path:
    """
    If prefix_or_path is an existing directory with results.json, return it.
    Else, treat it as a prefix and pick the most recent ./experiments/<prefix>_* directory.
    """
    p = Path(prefix_or_path)
    if p.is_dir() and (p / "results.json").exists():
        return p

    # search by prefix
    cand = []
    for d in (root).iterdir():
        if d.is_dir() and d.name.startswith(prefix_or_path) and (d / "results.json").exists():
            cand.append((d.stat().st_mtime, d))
    if not cand:
        raise FileNotFoundError(f"No experiment folder matches prefix '{prefix_or_path}' under {root}")
    cand.sort(key=lambda t: t[0], reverse=True)
    return cand[0][1]

def build_model(name, device):
    if name=="resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity(); m.fc = nn.Linear(512,10)
    elif name=="resnet34":
        m = torchvision.models.resnet34(weights=None)
        m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity(); m.fc = nn.Linear(512,10)
    elif name=="mobilenet_v2":
        m = torchvision.models.mobilenet_v2(weights=None)
        m.features[0][0] = nn.Conv2d(3,32,3,1,1,bias=False); m.classifier[1] = nn.Linear(m.last_channel,10)
    else:
        raise ValueError(f"Unknown model_name: {name}")
    return m.to(device)

def load_model(expdir: Path, device):
    rj = expdir / "results.json"
    if not rj.exists():
        raise FileNotFoundError(f"Missing results.json in {expdir}")
    with rj.open() as f:
        meta = json.load(f)
    cfg = Config()
    cfg.__dict__.update(meta.get("config", {}))
    ckpt_path = expdir / "best_model.pth"
    if not ckpt_path.exists():
        # fallback to any checkpoint
        cands = sorted(expdir.glob("checkpoint_epoch_*.pth"))
        if not cands:
            raise FileNotFoundError(f"No checkpoint found in {expdir}")
        ckpt_path = cands[-1]
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg.model_name, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg

@torch.no_grad()
def eval_clean(model, loader, device, normalized, max_batches=0):
    total=0; correct=0; seen_batches=0
    for x,y in loader:
        if max_batches and seen_batches>=max_batches: break
        x,y = x.to(device), y.to(device)
        logits = model(Attacks.normalize(x) if normalized else x)
        correct += (logits.argmax(1)==y).sum().item()
        total += y.size(0)
        seen_batches += 1
    return correct/total

def eval_attack(model, loader, device, normalized, steps, eps, step_size, max_batches=0):
    total=0; correct=0; seen_batches=0
    for x,y in loader:
        if max_batches and seen_batches>=max_batches: break
        x,y = x.to(device), y.to(device)
        # IMPORTANT: ensure grads are enabled for crafting
        torch.set_grad_enabled(True)
        x_adv = Attacks.pgd(model, x, y, eps, step_size, steps, str(device),
                            normalized=normalized, set_eval=True)
        torch.set_grad_enabled(False)
        with torch.no_grad():
            logits = model(Attacks.normalize(x_adv) if normalized else x_adv)
            correct += (logits.argmax(1)==y).sum().item()
        total += y.size(0)
        seen_batches += 1
    return correct/total

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir_a", required=True,
                    help="Prefix or full path for model A (e.g., heap-rat040_030-B1000k-resnet18)")
    ap.add_argument("--expdir_b", required=True,
                    help="Prefix or full path for model B (baseline, e.g., no_heap-B1000k-resnet18)")
    ap.add_argument("--experiments_root", default="./experiments")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--max_samples", type=int, default=10000, help="cap total images evaluated")
    ap.add_argument("--attacks", default="standard", choices=["standard"])
    ap.add_argument("--out_dir", default="./aa_out")
    args = ap.parse_args()

    root = Path(args.experiments_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # resolve prefixes to latest dirs
    A_dir = resolve_latest_dir(args.expdir_a, root)
    B_dir = resolve_latest_dir(args.expdir_b, root)

    dev = torch.device(args.device)
    modelA, cfgA = load_model(A_dir, dev)
    modelB, cfgB = load_model(B_dir, dev)

    # data loader from the "target" cfg (same CIFAR-10 either way)
    _, _, testloader = get_cifar10_loaders(cfgB)

    # bound batches by max_samples
    bs = testloader.batch_size if hasattr(testloader, "batch_size") else 128
    max_batches = (args.max_samples + bs - 1) // bs if args.max_samples>0 else 0

    # --- Evaluate A ---
    t0=time.time()
    cleanA = eval_clean(modelA, testloader, dev, cfgA.use_normalization, max_batches)
    fgsmA  = eval_attack(modelA, testloader, dev, cfgA.use_normalization, steps=1,  eps=8/255, step_size=2/255, max_batches=max_batches)
    pgd20A = eval_attack(modelA, testloader, dev, cfgA.use_normalization, steps=20, eps=8/255, step_size=2/255, max_batches=max_batches)
    pgd50A = eval_attack(modelA, testloader, dev, cfgA.use_normalization, steps=50, eps=8/255, step_size=2/255, max_batches=max_batches)
    wallA = time.time()-t0

    # --- Evaluate B ---
    t0=time.time()
    cleanB = eval_clean(modelB, testloader, dev, cfgB.use_normalization, max_batches)
    fgsmB  = eval_attack(modelB, testloader, dev, cfgB.use_normalization, steps=1,  eps=8/255, step_size=2/255, max_batches=max_batches)
    pgd20B = eval_attack(modelB, testloader, dev, cfgB.use_normalization, steps=20, eps=8/255, step_size=2/255, max_batches=max_batches)
    pgd50B = eval_attack(modelB, testloader, dev, cfgB.use_normalization, steps=50, eps=8/255, step_size=2/255, max_batches=max_batches)
    wallB = time.time()-t0

    summary = {
        "A_dir": str(A_dir),
        "B_dir": str(B_dir),
        "A": {
            "model": cfgA.model_name, "mode": cfgA.mode,
            "clean": cleanA, "fgsm": fgsmA, "pgd20": pgd20A, "pgd50": pgd50A,
            "wall_sec": wallA
        },
        "B": {
            "model": cfgB.model_name, "mode": cfgB.mode,
            "clean": cleanB, "fgsm": fgsmB, "pgd20": pgd20B, "pgd50": pgd50B,
            "wall_sec": wallB
        }
    }

    # print side-by-side
    def pct(x): return f"{100*x:.2f}%"
    print("\n== AutoEval (PGD-style) ==".upper())
    print(f"A: {A_dir.name}  |  B: {B_dir.name}")
    print(f"{'':12}  {'A (heap?)':>12}  {'B (baseline)':>14}")
    print(f"{'Clean':12}  {pct(cleanA):>12}  {pct(cleanB):>14}")
    print(f"{'FGSM':12}  {pct(fgsmA):>12}  {pct(fgsmB):>14}")
    print(f"{'PGD-20':12}  {pct(pgd20A):>12}  {pct(pgd20B):>14}")
    print(f"{'PGD-50':12}  {pct(pgd50A):>12}  {pct(pgd50B):>14}")
    print(f"{'Wall(s)':12}  {summary['A']['wall_sec']:>12.1f}  {summary['B']['wall_sec']:>14.1f}")

    # save reports
    tag = f"{A_dir.name}__VS__{B_dir.name}"
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / f"{tag}.txt").open("w") as f:
        f.write(f"A: {A_dir}\nB: {B_dir}\n")
        f.write(f"Clean A={cleanA:.4f} B={cleanB:.4f}\n")
        f.write(f"FGSM  A={fgsmA:.4f} B={fgsmB:.4f}\n")
        f.write(f"PGD20 A={pgd20A:.4f} B={pgd20B:.4f}\n")
        f.write(f"PGD50 A={pgd50A:.4f} B={pgd50B:.4f}\n")
        f.write(f"Wall  A={wallA:.1f}s B={wallB:.1f}s\n")
    print(f"\nSaved reports to {out_dir}/{tag}.json and .txt")
if __name__ == "__main__":
    main()
