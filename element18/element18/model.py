"""XGBoost training, threshold selection and calibration (Algorithms 5, 6)."""
import json
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .split import scoring_rows
from .features import feature_columns
from .metrics import row_metrics


def train_xgb(parts, features, device="cpu", seed=42, grid=None, rounds=400):
    tr, va = scoring_rows(parts["train"]), scoring_rows(parts["valid"])
    if tr["y"].sum() == 0 or va["y"].sum() == 0:
        raise ValueError("train or validation block has no positive rows")
    spw = float((tr["y"] == 0).sum() / max(1, (tr["y"] == 1).sum()))  # train only
    grid = grid or [{"max_depth": d, "learning_rate": lr, "min_child_weight": mcw}
                    for d in (3, 5) for lr in (0.05, 0.1) for mcw in (1, 5)]
    dtr = xgb.DMatrix(tr[features], label=tr["y"]); dva = xgb.DMatrix(va[features], label=va["y"])
    best = None
    for cand in grid:
        params = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
                  "device": device, "scale_pos_weight": spw, "seed": seed, "subsample": 0.8,
                  "colsample_bytree": 0.8, **cand}
        m = xgb.train(params, dtr, rounds, evals=[(dva, "valid")], early_stopping_rounds=30, verbose_eval=False)
        score = m.best_score
        if best is None or score > best[0]:
            best = (score, m, cand)
    return {"model": best[1], "params": best[2], "valid_aucpr": best[0], "scale_pos_weight": spw}


def predict(model, df, features):
    return model.predict(xgb.DMatrix(df[features]), iteration_range=(0, model.best_iteration + 1))


def choose_threshold(y, p, max_fa_note=None, beta=2.0):
    """Threshold maximising F-beta on the validation block (explicit recall emphasis)."""
    best_t, best = 0.5, -1
    for t in np.unique(np.round(np.quantile(p, np.linspace(0.5, 0.999, 200)), 6)):
        m = row_metrics(y, p, t)
        pr, rc = m["precision"], m["recall"]
        f = (1 + beta**2) * pr * rc / (beta**2 * pr + rc) if (beta**2 * pr + rc) > 0 else 0
        if f > best: best, best_t = f, float(t)
    return best_t


def fit_calibrator(y, p, method="isotonic"):
    if method == "isotonic" and y.sum() >= 30:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return ("isotonic", iso)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return ("platt", LogisticRegression().fit(z, y))


def apply_calibrator(cal, p):
    kind, m = cal
    if kind == "isotonic": return m.predict(p)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return m.predict_proba(z)[:, 1]


def save_model(model, path):
    """UBJSON/JSON files can be placed in a Triton FIL model repository."""
    model.save_model(path)
