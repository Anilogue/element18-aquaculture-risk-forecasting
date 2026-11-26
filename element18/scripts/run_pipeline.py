"""End-to-end run: data -> features -> labels -> split -> train -> evaluate -> explain.

Usage: python scripts/run_pipeline.py --config configs/exp_001.yaml [--csv path] [--device cuda]
Without --csv it uses SYNTHETIC data (code-path test only; says nothing about real ponds).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, yaml
from element18 import data, labels, features, split, model, metrics, baselines, explain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_001.yaml")
    ap.add_argument("--csv"); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seed = cfg["seed"]; np.random.seed(seed)
    if a.csv:
        df, rej = data.load_csv(a.csv, cfg["data"].get("freq_minutes", 60)); print("rejected rows:", len(rej))
        src = a.csv
    else:
        df = data.synthetic(seed=seed); src = "SYNTHETIC"
    L = cfg["label"]
    df = features.build_features(df)
    df = labels.add_labels(df, L["theta_mg_l"], L["k_samples"], L["horizon_h"], cfg["data"].get("freq_minutes", 60))
    parts = split.chrono_split(df, L["horizon_h"], frac=tuple(cfg["split"]["frac"]))
    feats = features.feature_columns(df)
    for k, v in parts.items():
        s = split.scoring_rows(v); print(f"{k}: rows={len(v)} scoring={len(s)} positives={int(s['y'].sum())}")
    device = a.device or cfg["model"].get("device", "cpu")
    fit = model.train_xgb(parts, feats, device=device, seed=seed)
    m = fit["model"]
    va, ca, te = (split.scoring_rows(parts[k]) for k in ("valid", "calib", "test"))
    pv, pc, pt = (model.predict(m, x, feats) for x in (va, ca, te))
    thr = model.choose_threshold(va["y"].to_numpy(), pv)
    cal = model.fit_calibrator(ca["y"].to_numpy(), pc)
    pt_cal = model.apply_calibrator(cal, pt)
    out = {"data_source": src, "config": cfg, "versions": {"numpy": np.__version__},
           "best_params": fit["params"], "threshold": thr, "calibrator": cal[0],
           "test_row_metrics": metrics.row_metrics(te["y"], pt, thr),
           "test_row_metrics_calibrated_brier": float(metrics.brier_score_loss(te["y"], pt_cal)) if hasattr(metrics, "brier_score_loss") else None,
           "test_event_metrics": metrics.event_metrics(te, pt, thr, L["horizon_h"]),
           "baseline_threshold_rule": metrics.row_metrics(te["y"], baselines.threshold_rule_score(te), 0.3),
           "baseline_logreg": metrics.row_metrics(te["y"], baselines.fit_logreg(parts, feats).predict_proba(te[feats])[:, 1], 0.5)}
    if len(te) and te["y"].sum():
        i = int(np.argmax(pt)); out["example_explanation"] = explain.explain_row(m, te.iloc[[i]], feats)
    os.makedirs(cfg["logging"]["out"], exist_ok=True)
    json.dump(out, open(os.path.join(cfg["logging"]["out"], "metrics.json"), "w"), indent=2, default=float)
    m.save_model(os.path.join(cfg["logging"]["out"], "model.json"))
    print(json.dumps({k: out[k] for k in ("data_source", "threshold", "test_row_metrics", "test_event_metrics")}, indent=2, default=float))
    print("NOTE: numbers from SYNTHETIC data must not be reported as results." if src == "SYNTHETIC" else "")

if __name__ == "__main__":
    main()
