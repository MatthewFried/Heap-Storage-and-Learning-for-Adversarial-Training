#!/usr/bin/env python3
import argparse, json, os, subprocess, sys, time, csv
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
LEARN = str(HERE / "learning_heap.py")

def run_one(tag, args_list):
    p = subprocess.run([PY, LEARN] + args_list,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, check=False)
    expdir = None
    for line in p.stdout.splitlines():
        if "Results saved to" in line:
            expdir = line.split("Results saved to", 1)[1].strip()
    ok = (p.returncode == 0 and expdir and Path(expdir).exists())
    return ok, expdir, p.stdout

def read_result(expdir):
    j = json.loads((Path(expdir) / "results.json").read_text())
    tm = j.get("test_metrics", {})
    em = j.get("efficiency_metrics", {})
    pgd_train = max(1, int(em.get("pgd_calls_train", 0)))
    robust = float(tm.get("robust_acc", 0.0))
    eff = robust * (10000.0 / pgd_train)
    return {
        "clean": tm.get("clean_acc"),
        "robust": robust,
        "pgd_train": em.get("pgd_calls_train"),
        "wall_s": j.get("total_wall_time"),
        "efficiency": eff,
        "expdir": expdir
    }

def main():
    ap = argparse.ArgumentParser(description="Heap vs No-Heap robustness vs PGD budget")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--pgd_steps", type=int, default=5)
    ap.add_argument("--model_name", default="resnet18", choices=["resnet18","resnet34"])
    ap.add_argument("--heap_ratio", type=float, default=0.4)
    ap.add_argument("--global_heap_ratio", type=float, default=0.3)
    ap.add_argument("--focus_mode", default="none", choices=["none","burst"])
    ap.add_argument("--burst_len", type=int, default=200)
    ap.add_argument("--initial_bank_size", type=int, default=1000)
    ap.add_argument("--out_dir", default="./budget_out")
    ap.add_argument("--start_k", type=int, default=300)  # thousands
    ap.add_argument("--end_k", type=int, default=1000)   # thousands
    ap.add_argument("--step_k", type=int, default=100)   # thousands
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "experiments").mkdir(parents=True, exist_ok=True)

    budgets = list(range(args.start_k*1000, args.end_k*1000 + 1, args.step_k*1000))
    rows = []

    common = [
        "--data_root", args.data_root,
        "--device", args.device,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--warmup_epochs", str(args.warmup),
        "--batch_size", str(args.batch_size),
        "--subset_size", "0",
        "--pgd_steps", str(args.pgd_steps),
        "--eval_pgd_steps", "20",
        "--eval_pgd_restarts", "2",
        "--model_name", args.model_name,
        "--output_dir", str(out_dir / "experiments"),
    ]

    for B in budgets:
        for mode in ["heap", "no_heap"]:
            tag = f"{mode}-B{B//1000}k-{args.model_name}"
            args_list = common + [
                "--experiment_name", tag,
                "--mode", mode,
                "--pgd_budget_train", str(B),
            ]
            if mode == "heap":
                args_list += [
                    "--heap_ratio", str(args.heap_ratio),
                    "--global_heap_ratio", str(args.global_heap_ratio),
                    "--focus_mode", args.focus_mode,
                    "--burst_len", str(args.burst_len),
                    "--initial_bank_size", str(args.initial_bank_size),
                ]

            t0 = time.time()
            ok, expdir, out = run_one(tag, args_list)
            if not ok:
                print(f"[WARN] failed: {tag}\n{out[-800:]}")
                continue
            r = read_result(expdir)
            r.update({"mode": mode, "budget": B, "tag": tag, "secs": round(time.time()-t0,1)})
            rows.append(r)
            print(f"[{tag}] robust={r['robust']:.4f} clean={r['clean']:.4f} "
                  f"pgd={r['pgd_train']} eff={r['efficiency']:.4f} wall={r['wall_s']:.1f}s")

    # CSV
    csv_path = out_dir / "budget_curve.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode","budget","robust","clean","pgd_train","efficiency","wall_s","tag","expdir","secs"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        H = sorted([r for r in rows if r["mode"]=="heap"], key=lambda r: r["budget"])
        N = sorted([r for r in rows if r["mode"]=="no_heap"], key=lambda r: r["budget"])

        plt.figure(figsize=(6.5,4.2))
        plt.plot([r["budget"]/1000 for r in H], [r["robust"] for r in H],
                 marker="o", label=f"heap (ρ={args.heap_ratio}, ρg={args.global_heap_ratio}, {args.focus_mode})")
        plt.plot([r["budget"]/1000 for r in N], [r["robust"] for r in N],
                 marker="s", label="no_heap")
        plt.xlabel("PGD Budget (thousands of calls)")
        plt.ylabel("Test Robust Accuracy")
        plt.title(f"Robustness vs Budget — {args.model_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        png = out_dir / "budget_curve.png"
        plt.tight_layout(); plt.savefig(png, dpi=160)
        print(f"Saved plot to {png}")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}. CSV is available for external plotting.")

if __name__ == "__main__":
    main()
