#!/usr/bin/env python3
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import torch

PY = sys.executable
HERE = Path(__file__).resolve().parent
LEARN = str(HERE / "learning_heap.py")

# --- helpers ---
def run_one(tag, args_list, env=None):
    t0 = time.time()
    expdir, results_json = None, None
    ok = True
    try:
        p = subprocess.run([PY, LEARN] + args_list,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=env, check=False)
        for line in p.stdout.splitlines():
            if "Results saved to" in line:
                expdir = line.split("Results saved to",1)[1].strip()
        if p.returncode != 0 or expdir is None:
            ok = False
    except Exception as e:
        ok = False
        p = type("obj", (), {"stdout": str(e)})

    row = {"tag": tag, "ok": ok, "wall_s": round(time.time()-t0, 2),
           "stdout_tail": "\n".join(p.stdout.splitlines()[-20:])}
    if ok and expdir:
        rj = Path(expdir) / "results.json"
        if rj.exists():
            with open(rj) as f:
                results_json = json.load(f)
            cfg = results_json.get("config", {})
            tm = results_json.get("test_metrics", {})
            em = results_json.get("efficiency_metrics", {})
            row.update({
                "experiment_dir": expdir,
                "mode": cfg.get("mode"),
                "model": cfg.get("model_name","resnet18"),
                "seed": cfg.get("seed"),
                "epochs": cfg.get("epochs"),
                "pgd_steps": cfg.get("pgd_steps"),
                "heap_ratio": cfg.get("heap_ratio"),
                "global_heap_ratio": cfg.get("global_heap_ratio"),
                "focus_mode": cfg.get("focus_mode","none"),
                "budget": cfg.get("pgd_budget_train"),
                "trades_beta": cfg.get("trades_beta"),
                "test_clean": tm.get("clean_acc"),
                "test_robust": tm.get("robust_acc"),
                "best_val_robust": results_json.get("best_val_robust"),
                "efficiency": results_json.get("efficiency_score"),
                "total_wall": results_json.get("total_wall_time"),
                "pgd_calls_train": em.get("pgd_calls_train"),
                "pgd_calls_eval": em.get("pgd_calls_eval"),
                "unique_variants": em.get("total_unique_variants"),
            })
    return row, results_json

def append_summary(row, summary_path):
    if not row.get("ok"): return
    line = (
        f"[{row.get('tag')}] "
        f"mode={row.get('mode')} model={row.get('model')} "
        f"budget={row.get('budget')} clean={row.get('test_clean'):.4f} "
        f"robust={row.get('test_robust'):.4f} eff={row.get('efficiency'):.4f} "
        f"pgd_train={row.get('pgd_calls_train')} wall={row.get('total_wall'):.1f}s"
    )
    with open(summary_path,"a") as f: f.write(line+"\n")

def append_jsonl(row, results_json, jsonl_path):
    if not row.get("ok") or results_json is None: return
    payload = {
        "tag": row.get("tag"),
        "experiment_dir": row.get("experiment_dir"),
        "checkpoint": "best_model.pth",
        "config": results_json.get("config", {}),
        "test_metrics": results_json.get("test_metrics", {}),
        "efficiency_metrics": results_json.get("efficiency_metrics", {}),
    }
    with open(jsonl_path,"a") as f: f.write(json.dumps(payload)+"\n")

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="suite_compact.csv")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pgd_steps", type=int, default=5)
    args = ap.parse_args()

    summary_path = "suite_summary.txt"
    jsonl_path   = "transfer_input.jsonl"
    open(summary_path,"w").close()
    open(jsonl_path,"w").close()

    # common args
    common = [
        "--data_root", args.data_root,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--warmup_epochs", str(args.warmup),
        "--pgd_steps", str(args.pgd_steps),
        "--device", args.device,
        "--subset_size", "0",
        "--eval_pgd_steps", "20",
        "--eval_pgd_restarts", "2",
        "--pgd_budget_train", "1000000",
    ]

    rows = []

    # Core 6
    for model in ["resnet18","mobilenet_v2"]:
        for mode in ["heap","no_heap","random_buffer"]:
            tag=f"{mode}-B1000k-{model}"
            args_list = common + ["--experiment_name",tag,"--mode",mode,"--model_name",model]
            row,rj=run_one(tag,args_list); rows.append(row)
            append_summary(row,summary_path); append_jsonl(row,rj,jsonl_path)

    # Focus 4
    for model in ["resnet18","mobilenet_v2"]:
        for fm in ["none","burst"]:
            tag=f"heap-{fm}-B1000k-{model}"
            args_list=common+["--experiment_name",tag,"--mode","heap","--model_name",model,"--focus_mode",fm]
            row,rj=run_one(tag,args_list); rows.append(row)
            append_summary(row,summary_path); append_jsonl(row,rj,jsonl_path)

    # TRADES 2
    for mode in ["heap","no_heap"]:
        tag=f"{mode}-B1000k-resnet18-tb0.3"
        args_list=common+["--experiment_name",tag,"--mode",mode,"--model_name","resnet18","--trades_beta","0.3"]
        row,rj=run_one(tag,args_list); rows.append(row)
        append_summary(row,summary_path); append_jsonl(row,rj,jsonl_path)

    # Ratios 3
    ratios=[("040_030",0.4,0.3),("030_020",0.3,0.2),("060_030",0.6,0.3)]
    for name,hr,gr in ratios:
        tag=f"heap-rat{name}-B1000k-resnet18"
        args_list=common+["--experiment_name",tag,"--mode","heap","--model_name","resnet18",
                          "--heap_ratio",str(hr),"--global_heap_ratio",str(gr)]
        row,rj=run_one(tag,args_list); rows.append(row)
        append_summary(row,summary_path); append_jsonl(row,rj,jsonl_path)

    # Cheaper PGD 2
    for model in ["resnet18","mobilenet_v2"]:
        tag=f"heap-pgd3-B1000k-{model}"
        args_list=[*common,"--pgd_steps","3","--experiment_name",tag,"--mode","heap","--model_name",model]
        row,rj=run_one(tag,args_list); rows.append(row)
        append_summary(row,summary_path); append_jsonl(row,rj,jsonl_path)

    # FIFO 1
    tag="fifo-B1000k-resnet18"
    args_list=common+["--experiment_name",tag,"--mode","fifo","--model_name","resnet18"]
    row,rj=run_one(tag,args_list); rows.append(row)
    append_summary(row,summary_path); append_jsonl(row,rj,jsonl_path)

    # Resnet34 2
    for mode in ["heap","no_heap"]:
        tag=f"{mode}-B1000k-resnet34"
        args_list=common+["--experiment_name",tag,"--mode",mode,"--model_name","resnet34"]
        row,rj=run_one(tag,args_list); rows.append(row)
        append_summary(row,summary_path); append_jsonl(row,rj,jsonl_path)

    # --- write CSV ---
    out=Path(args.csv)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r.keys()}))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} with {len(rows)} rows. Summary in {summary_path}, JSONL in {jsonl_path}")

if __name__=="__main__":
    main()
