#!/usr/bin/env python3
import argparse, csv, json, os, re, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torchvision

# --- import bits we already have in your project ---
from learning_heap import Attacks, get_cifar10_loaders, Config

# ----------------- helpers -----------------
def build_model(name: str, device: torch.device) -> nn.Module:
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(512, 10)
    elif name == "resnet34":
        m = torchvision.models.resnet34(weights=None)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(512, 10)
    else:
        raise ValueError(f"Unsupported model for transfer: {name}")
    return m.to(device)

def load_result(exp_dir: Path) -> Dict[str, Any]:
    rj = exp_dir / "results.json"
    if not rj.exists():
        return {}
    with rj.open() as f:
        j = json.load(f)
    return j

def load_model_from_exp(exp_dir: Path, device: torch.device) -> Tuple[nn.Module, Config]:
    j = load_result(exp_dir)
    if not j:
        raise RuntimeError(f"No results.json in {exp_dir}")
    cfg = Config()  # start with defaults then override from file
    cfg.__dict__.update(j.get("config", {}))
    if cfg.model_name not in ("resnet18", "resnet34"):
        raise RuntimeError(f"Filtered-out or unsupported model: {cfg.model_name}")

    ckpt_path = exp_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise RuntimeError(f"No best_model.pth in {exp_dir}")
    state = torch.load(ckpt_path, map_localtion=device) if hasattr(torch.load, "map_localtion") else torch.load(ckpt_path, map_location=device)

    m = build_model(cfg.model_name, device)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    return m, cfg

def iter_experiments(root: Path) -> List[Path]:
    # experiment folders contain results.json
    out = []
    for p in root.glob("*"):
        if p.is_dir() and (p / "results.json").exists():
            out.append(p)
    return sorted(out)

def pick_best_by_mode_and_model(exps: List[Path]) -> Dict[Tuple[str, str], Path]:
    """
    Return best expdir per (mode, model_name), measured by test robust_acc.
    """
    best: Dict[Tuple[str, str], Tuple[float, Path]] = {}
    for e in exps:
        j = load_result(e)
        if not j:
            continue
        cfg = j.get("config", {})
        mode = cfg.get("mode", "")
        model = cfg.get("model_name", "")
        if model == "mobilenet_v2":  # filter out MobileNet
            continue
        tm = j.get("test_metrics", {})
        robust = float(tm.get("robust_acc", 0.0))
        key = (mode, model)
        if key not in best or robust > best[key][0]:
            best[key] = (robust, e)
    return {k: v for k, (_, v) in best.items()}

def planned_pairs(best: Dict[Tuple[str, str], Path]) -> List[Tuple[Path, Path]]:
    """
    Build the planned transfer pairs using only:
      - resnet18 heap <-> no_heap
      - resnet34 heap <-> no_heap
      - cross-arch: r18(heap)->r34(no_heap), r34(heap)->r18(no_heap),
                    r18(no_heap)->r34(heap), r34(no_heap)->r18(heap)
    Missing entries are skipped automatically.
    """
    pairs: List[Tuple[Path, Path]] = []

    def has(mode, model): return (mode, model) in best

    # within-arch
    for model in ("resnet18", "resnet34"):
        if has("heap", model) and has("no_heap", model):
            pairs.append((best[("heap", model)], best[("no_heap", model)]))
            pairs.append((best[("no_heap", model)], best[("heap", model)]))

    # cross-arch
    # r18(heap) -> r34(no_heap) and reverse
    if has("heap", "resnet18") and has("no_heap", "resnet34"):
        pairs.append((best[("heap", "resnet18")], best[("no_heap", "resnet34")]))
    if has("heap", "resnet34") and has("no_heap", "resnet18"):
        pairs.append((best[("heap", "resnet34")], best[("no_heap", "resnet18")]))

    # r18(no_heap) -> r34(heap) and reverse
    if has("no_heap", "resnet18") and has("heap", "resnet34"):
        pairs.append((best[("no_heap", "resnet18")], best[("heap", "resnet34")]))
    if has("no_heap", "resnet34") and has("heap", "resnet18"):
        pairs.append((best[("no_heap", "resnet34")], best[("heap", "resnet18")]))

    # dedup
    seen = set()
    uniq = []
    for s, t in pairs:
        key = (str(s), str(t))
        if key not in seen:
            seen.add(key)
            uniq.append((s, t))
    return uniq

@torch.no_grad()
def clean_acc(model: nn.Module, loader, normalized: bool) -> float:
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        logits = model(Attacks.normalize(x) if normalized else x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct/total

def transfer_once(src_exp: Path, tgt_exp: Path, device: torch.device,
                  steps: int, eps: float, step_size: float, max_batches: int = 0) -> Dict[str, Any]:
    # Load both models and test loader (use target config’s normalization / split)
    src_model, src_cfg = load_model_from_exp(src_exp, device)
    tgt_model, tgt_cfg = load_model_from_exp(tgt_exp, device)
    _, _, testloader = get_cifar10_loaders(tgt_cfg)

    total = 0
    robust = 0

    # IMPORTANT: PGD must be run with grads enabled (no torch.no_grad here).
    batches = 0
    for x, y in testloader:
        if max_batches and batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        # craft on SOURCE (using SOURCE normalization flag)
        x_adv = Attacks.pgd(
            src_model, x, y, eps, step_size, steps,
            str(device), normalized=src_cfg.use_normalization, set_eval=True
        )

        # evaluate on TARGET (using TARGET normalization flag)
        with torch.no_grad():
            logits = tgt_model(Attacks.normalize(x_adv) if tgt_cfg.use_normalization else x_adv)
            robust += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        batches += 1

    return {
        "source_dir": str(src_exp),
        "target_dir": str(tgt_exp),
        "source_mode": src_cfg.mode,
        "target_mode": tgt_cfg.mode,
        "source_model": src_cfg.model_name,
        "target_model": tgt_cfg.model_name,
        "steps": steps,
        "eps": eps,
        "step_size": step_size,
        "transfer_robust_acc": robust / max(1, total),
        "target_clean_acc": None  # can fill if needed
    }

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="./experiments")
    ap.add_argument("--out_dir", default="./transfer_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eps", type=float, default=8/255)
    ap.add_argument("--step_size", type=float, default=2/255)
    ap.add_argument("--max_batches", type=int, default=0, help="0 => all test batches")
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_root = Path(args.exp_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exps = iter_experiments(exp_root)
    if not exps:
        print(f"No experiments found under {exp_root}", file=sys.stderr)
        sys.exit(1)

    # pick best per (mode, model) after filtering out MobileNet
    best = pick_best_by_mode_and_model(exps)

    # announce what we found
    print("Best runs per (mode, model):")
    for (mode, model), p in sorted(best.items()):
        j = load_result(p)
        tm = j.get("test_metrics", {})
        print(f"  {(mode, model)} -> {p.name} | robust={tm.get('robust_acc'):.4f}")

    pairs = planned_pairs(best)
    if not pairs:
        print("No valid transfer pairs after filtering; exiting.")
        sys.exit(0)

    print("\nPlanned transfer pairs:")
    for s, t in pairs:
        print(f"  {s.name}  -->  {t.name}")

    # outputs
    csv_path = out_dir / "transfer_summary.csv"
    jsonl_path = out_dir / "transfer_results.jsonl"

    rows = []
    with jsonl_path.open("w") as jf:
        for s, t in pairs:
            res = transfer_once(s, t, device, args.steps, args.eps, args.step_size, args.max_batches)
            # pretty one-liner
            print(f"[TRANSFER] {Path(res['source_dir']).name} -> {Path(res['target_dir']).name} "
                  f"| steps={res['steps']} eps={res['eps']:.5f} "
                  f"| transfer_robust={res['transfer_robust_acc']:.4f}")
            jf.write(json.dumps(res) + "\n")
            rows.append({
                "source": Path(res["source_dir"]).name,
                "target": Path(res["target_dir"]).name,
                "src_mode": res["source_mode"],
                "tgt_mode": res["target_mode"],
                "src_model": res["source_model"],
                "tgt_model": res["target_model"],
                "steps": res["steps"],
                "eps": res["eps"],
                "step_size": res["step_size"],
                "transfer_robust_acc": f"{res['transfer_robust_acc']:.4f}",
            })

    # write CSV
    if rows:
        keys = ["source","target","src_mode","tgt_mode","src_model","tgt_model",
                "steps","eps","step_size","transfer_robust_acc"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote CSV to {csv_path}")
        print(f"Wrote JSONL to {jsonl_path}")
    else:
        print("No rows written (no pairs?)")

if __name__ == "__main__":
    main()
