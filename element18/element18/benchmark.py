"""CPU/GPU benchmark harness (Algorithm 13). Reports measured values only."""
import json, statistics, subprocess, time
import numpy as np
import pandas as pd
from .data import synthetic
from .features import build_core_features


def _gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,temperature.gpu,clocks.sm,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"


def _sync():
    try:
        import cupy; cupy.cuda.Stream.null.synchronize()
    except Exception:
        pass


def time_op(fn, trials=7, warmup=2, sync=False):
    for _ in range(warmup):
        fn(); sync and _sync()
    ts = []
    for _ in range(trials):
        t = time.perf_counter(); fn(); sync and _sync(); ts.append(time.perf_counter() - t)
    return {"median_s": statistics.median(ts), "mean_s": statistics.mean(ts),
            "sd_s": statistics.pstdev(ts), "min_s": min(ts), "max_s": max(ts), "trials": trials}


def make_workload(rows, ponds=10):
    per = max(1, rows // ponds)
    days = max(1, per // 24 + 1)
    df = synthetic(n_ponds=ponds, days=days)
    return df.iloc[:rows].copy()


def run_preprocessing(sizes=(10_000, 100_000, 1_000_000), trials=7, out="results/bench_preproc.json"):
    results = []
    try:
        import cudf
        have_gpu = True
    except Exception:
        have_gpu = False
    for n in sizes:
        pdf = make_workload(n)
        rec = {"rows": len(pdf), "gpu_info_before": _gpu_info()}
        rec["pandas"] = time_op(lambda: build_core_features(pdf), trials)
        if have_gpu:
            t = time.perf_counter(); gdf = cudf.from_pandas(pdf); _sync()
            rec["transfer_h2d_s"] = time.perf_counter() - t
            rec["cudf"] = time_op(lambda: build_core_features(gdf), trials, sync=True)
            rec["speedup_median"] = rec["pandas"]["median_s"] / rec["cudf"]["median_s"]
        else:
            rec["cudf"] = "not run: cudf not installed or no GPU"
        rec["gpu_info_after"] = _gpu_info()
        results.append(rec)
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    return results


def run_xgb(df_parts, features, trials=3, out="results/bench_xgb.json"):
    import xgboost as xgb
    from .split import scoring_rows
    tr = scoring_rows(df_parts["train"]); te = scoring_rows(df_parts["test"])
    res = {}
    for dev in ("cpu", "cuda"):
        try:
            params = {"objective": "binary:logistic", "tree_method": "hist", "device": dev, "max_depth": 5}
            d = xgb.DMatrix(tr[features], label=tr["y"])
            def train(): return xgb.train(params, d, 200)
            res[dev] = {"train": time_op(train, trials, warmup=1)}
            m = train(); dt = xgb.DMatrix(te[features])
            res[dev]["batch_predict"] = time_op(lambda: m.predict(dt), 7)
            one = xgb.DMatrix(te[features].iloc[:1])
            res[dev]["single_row_predict"] = time_op(lambda: m.predict(one), 50, warmup=5)
        except Exception as e:
            res[dev] = f"not run: {type(e).__name__}: {e}"
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    return res


def run_llm_latency(inp, model="llama3.1:8b", url="http://localhost:11434", trials=5, out="results/bench_llm.json"):
    import requests
    from .advisory import SYSTEM_PROMPT
    rows = []
    for i in range(trials):
        t = time.perf_counter()
        r = requests.post(f"{url}/api/chat", timeout=300, json={
            "model": model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(inp)}]}).json()
        wall = time.perf_counter() - t
        rows.append({"wall_s": wall, "load_s": r.get("load_duration", 0) / 1e9,
                     "prompt_eval_s": r.get("prompt_eval_duration", 0) / 1e9,
                     "eval_s": r.get("eval_duration", 0) / 1e9, "eval_tokens": r.get("eval_count"),
                     "tokens_per_s": (r.get("eval_count", 0) / (r.get("eval_duration", 1) / 1e9)),
                     "first_call_includes_load": i == 0})
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    return rows
