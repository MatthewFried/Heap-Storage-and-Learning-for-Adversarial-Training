#!/usr/bin/env python3
"""
Run ablations for Heap-Guided Robust Training.
- Varies one knob at a time around a good baseline.
- Parses results.json from each run.
- Produces ablation_summary.csv and simple plots.

Usage examples:
  # Quick smoke on 10k subset, fewer epochs
  python run_all_ablations.py --script heap_hex.py --epochs 10 --subset 10000

  # Full CIFAR-10 (50k), 30 epochs
  python run_all_ablations.py --script heap_hex.py --epochs 30 --subset 0

  # Resume from a specific section (e.g., bucket_strategy)
  python run_all_ablations.py --script heap_hex.py --epochs 30 --subset 10000 --start_from bucket_strategy
"""
import argparse, subprocess, time, json, os, sys, csv
from pathlib import Path
from datetime import datetime
import itertools

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    return proc.returncode

def find_latest_results(experiments_dir: Path, started_after: float):
    """Return (experiment_dir, results_dict) for the newest results.json created after started_after."""
    newest = None
    for p in experiments_dir.glob("*"):
        rj = p / "results.json"
        if rj.exists():
            mtime = rj.stat().st_mtime
            if mtime >= started_after - 2:  # small slack
                if newest is None or mtime > newest[0]:
                    newest = (mtime, p, rj)
    if newest is None:
        return None, None
    with open(newest[2], "r") as f:
        data = json.load(f)
    return newest[1], data

def extract_row(sweep, label, cfg_overrides, results):
    eff = results.get("efficiency_score", None)
    em = results.get("efficiency_metrics", {})
    tm = results.get("test_metrics", {})
    cfg = results.get("config", {})
    last_entropy = (em.get("bucket_entropy") or [None])[-1]
    # class coverage is a list of dicts; take last snapshot if present
    last_cov = (em.get("per_class_coverage") or [{}])[-1]
    coverage_nonzero = len([k for k,v in last_cov.items() if v > 0]) if isinstance(last_cov, dict) else None
    return {
        "sweep": sweep,
        "label": label,
        "mode": cfg.get("mode"),
        "heap_ratio": cfg.get("heap_ratio"),
        "heap_size": cfg.get("heap_max_size"),
        "bucket_strategy": cfg.get("bucket_strategy"),
        "diversity_k_factor": cfg.get("diversity_k_factor"),
        "pgd_steps_train": cfg.get("pgd_steps"),
        "eval_pgd_steps": cfg.get("eval_pgd_steps"),
        "eval_pgd_restarts": cfg.get("eval_pgd_restarts"),
        "epochs": cfg.get("epochs"),
        "subset_size": cfg.get("subset_size"),
        "val_split": cfg.get("val_split"),
        "warmup_epochs": cfg.get("warmup_epochs"),
        "clean_acc": tm.get("clean_acc"),
        "robust_acc": tm.get("robust_acc"),
        "pgd_calls_train": em.get("pgd_calls_train"),
        "efficiency_score": eff,
        "bucket_entropy_last": last_entropy,
        "class_coverage_nonzero_last": coverage_nonzero,
    }

def make_plot_xy(rows, xkey, ykeys, outpath, title):
    if not HAVE_MPL:
        return
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        xs = [r[xkey] for r in rows]
        for yk in ykeys:
            ys = [r[yk] for r in rows]
            plt.plot(xs, ys, marker="o", label=yk)
        plt.xlabel(xkey)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outpath, dpi=140)
        plt.close()
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="heap_hex.py", help="Path to heap_hex.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--subset", type=int, default=0, help="0 means full CIFAR-10 train set")
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--heap_size", type=int, default=2000)
    ap.add_argument("--bucket_strategy", default="class_hardness",
                        choices=["class","class_hardness","kmeans","tree"])
    ap.add_argument("--heap_ratio", type=float, default=0.5)
    ap.add_argument("--diversity_k_factor", type=int, default=3)
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--eval_pgd_steps", type=int, default=20)
    ap.add_argument("--eval_pgd_restarts", type=int, default=2)
    ap.add_argument("--experiment_name", default="ablation")
    ap.add_argument("--output", default="sweeps")
    ap.add_argument("--data_root", default="./data")  # not directly passed; heap_hex handles it internally

    # NEW: allow resuming from a specific sweep
    ap.add_argument("--start_from", default="all",
                    choices=["all","baselines","heap_ratio","heap_size","bucket_strategy","diversity_k_factor","pgd_steps"],
                    help="Begin running ablations from this section")
    args = ap.parse_args()

    here = Path.cwd()
    experiments_dir = here / "experiments"
    sweeps_dir = here / args.output
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Baseline used for "vary one thing" ----------
    baseline = dict(
        mode="heap",
        epochs=args.epochs,
        subset=args.subset,
        warmup_epochs=args.warmup_epochs,
        heap_size=args.heap_size,
        bucket_strategy=args.bucket_strategy,
        heap_ratio=args.heap_ratio,
        diversity_k_factor=args.diversity_k_factor,
        pgd_steps=args.pgd_steps,
        eval_pgd_steps=args.eval_pgd_steps,
        eval_pgd_restarts=args.eval_pgd_restarts
    )

    # Helper to invoke a single run
    def run_one(tag, overrides):
        cfg = baseline.copy()
        cfg.update(overrides or {})
        # Build command line
        cmd = [
            args.python, args.script,
            "--experiment", "single",
            "--experiment_name", args.experiment_name,
            "--mode", str(cfg["mode"]),
            "--epochs", str(cfg["epochs"]),
            "--subset", str(cfg["subset"]),
            "--warmup_epochs", str(cfg["warmup_epochs"]),
            "--heap_size", str(cfg["heap_size"]),
            "--bucket_strategy", str(cfg["bucket_strategy"]),
            "--heap_ratio", str(cfg["heap_ratio"]),
            "--diversity_k_factor", str(cfg["diversity_k_factor"]),
            "--pgd_steps", str(cfg["pgd_steps"]),
            "--eval_pgd_steps", str(cfg["eval_pgd_steps"]),
            "--eval_pgd_restarts", str(cfg["eval_pgd_restarts"]),
        ]
        # No CHI flags -> stays disabled
        start = time.time()
        log_file = sweeps_dir / f"{tag}.log"
        rc = run(cmd, log_file)
        if rc != 0:
            print(f"[{tag}] FAILED (see {log_file})")
            return None, None
        exp_dir, res = find_latest_results(experiments_dir, start)
        if res is None:
            print(f"[{tag}] Could not find results.json; see {log_file}")
        return exp_dir, res

    rows = []

    # NEW: gating helper for resuming at a section
    steps_order = ["baselines","heap_ratio","heap_size","bucket_strategy","diversity_k_factor","pgd_steps"]
    start_idx = 0 if args.start_from == "all" else steps_order.index(args.start_from)
    def should_run(step_name: str) -> bool:
        return steps_order.index(step_name) >= start_idx

    # 1) Baselines: no_heap and heap (baseline)
    if should_run("baselines"):
        print("Running baselines...")
        for mode in ["no_heap", "heap"]:
            tag = f"baseline_{mode}"
            exp_dir, res = run_one(tag, {"mode": mode})
            if res: rows.append(extract_row("baseline", tag, {}, res))

    # 2) Sampling mix: heap_ratio ∈ {0.25, 0.5, 0.75}
    if should_run("heap_ratio"):
        print("Sweep: heap_ratio")
        for r in [0.25, 0.5, 0.75]:
            tag = f"ratio_{r}"
            exp_dir, res = run_one(tag, {"mode":"heap", "heap_ratio": r})
            if res: rows.append(extract_row("heap_ratio", tag, {"heap_ratio": r}, res))

    # 3) Heap size: {500, 1000, 2000}
    if should_run("heap_size"):
        print("Sweep: heap_size")
        for hs in [500, 1000, 2000]:
            tag = f"heapsize_{hs}"
            exp_dir, res = run_one(tag, {"mode":"heap", "heap_size": hs})
            if res: rows.append(extract_row("heap_size", tag, {"heap_size": hs}, res))

    # 4) Bucket strategy: class, class_hardness, kmeans, tree
    if should_run("bucket_strategy"):
        print("Sweep: bucket_strategy")
        for strat in ["class","kmeans","tree"]:
            tag = f"bucket_{strat}"
            exp_dir, res = run_one(tag, {"mode":"heap", "bucket_strategy": strat})
            if res: rows.append(extract_row("bucket_strategy", tag, {"bucket_strategy": strat}, res))

    # 5) Diversity factor: {3,5}
    if should_run("diversity_k_factor"):
        print("Sweep: diversity_k_factor")
        for dk in [3,5]:
            tag = f"divk_{dk}"
            exp_dir, res = run_one(tag, {"mode":"heap", "diversity_k_factor": dk})
            if res: rows.append(extract_row("diversity_k_factor", tag, {"diversity_k_factor": dk}, res))

    # 6) PGD steps: {5, 10} train; eval fixed at 20
    if should_run("pgd_steps"):
        print("Sweep: pgd_steps")
        for ps in [5,10]:
            tag = f"pgd_{ps}"
            exp_dir, res = run_one(tag, {"mode":"heap", "pgd_steps": ps})
            if res: rows.append(extract_row("pgd_steps", tag, {"pgd_steps": ps}, res))

    # Write CSV
    csv_path = sweeps_dir / "ablation_summary.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"Wrote {csv_path}")

    # Plots (optional)
    if HAVE_MPL and rows:
        # simple helpers to select subset by sweep
        def sel(sweep): return [r for r in rows if r["sweep"] == sweep]
        # efficiency & robustness vs heap_ratio
        make_plot_xy(sel("heap_ratio"), "heap_ratio",
                     ["robust_acc","efficiency_score"],
                     sweeps_dir / "heap_ratio.png",
                     "Sampling mix (heap_ratio) sweep")
        # vs heap_size
        make_plot_xy(sel("heap_size"), "heap_size",
                     ["robust_acc","efficiency_score"],
                     sweeps_dir / "heap_size.png",
                     "Heap size sweep")
        # vs diversity_k_factor
        make_plot_xy(sel("diversity_k_factor"), "diversity_k_factor",
                     ["robust_acc","efficiency_score"],
                     sweeps_dir / "diversity_k_factor.png",
                     "Diversity factor sweep")
        # vs pgd_steps
        make_plot_xy(sel("pgd_steps"), "pgd_steps_train",
                     ["robust_acc","efficiency_score"],
                     sweeps_dir / "pgd_steps.png",
                     "PGD steps (train) sweep")
        # bucket strategy: bar chart
        try:
            import matplotlib.pyplot as plt
            b = sel("bucket_strategy")
            if b:
                x = [r["bucket_strategy"] for r in b]
                y1 = [r["robust_acc"] for r in b]
                y2 = [r["efficiency_score"] for r in b]
                fig, ax1 = plt.subplots(figsize=(6,4))
                xs = range(len(x))
                ax1.bar([i-0.2 for i in xs], y1, width=0.4, label="robust_acc")
                ax2 = ax1.twinx()
                ax2.bar([i+0.2 for i in xs], y2, width=0.4, label="efficiency_score")
                ax1.set_xticks(list(xs)); ax1.set_xticklabels(x, rotation=15)
                ax1.set_title("Bucket strategy sweep")
                fig.tight_layout()
                plt.savefig(sweeps_dir / "bucket_strategy.png", dpi=140)
                plt.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
