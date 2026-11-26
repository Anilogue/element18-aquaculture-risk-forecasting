"""Row-level and event-level metrics."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, precision_recall_fscore_support)


def row_metrics(y, p, thr):
    y = np.asarray(y); p = np.asarray(p); pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    beta2 = 4.0
    f2 = (1 + beta2) * pr * rc / (beta2 * pr + rc) if (beta2 * pr + rc) > 0 else 0.0
    out = {"precision": pr, "recall": rc, "f1": f1, "f2": f2,
           "brier": brier_score_loss(y, p)}
    if len(np.unique(y)) == 2:
        out["pr_auc"] = average_precision_score(y, p)
        out["roc_auc"] = roc_auc_score(y, p)
    else:
        out["pr_auc"] = out["roc_auc"] = float("nan")
    return out


def event_metrics(df, p, thr, horizon_h=12, freq_minutes=60):
    """df: scoring rows with pond_id, timestamp, onset (full pond series needed for onsets).
    An onset is detected if an alert fired in the H hours before it.
    False alarm = alert with no onset within H after it (counted per pond-day).
    """
    h = int(round(horizon_h * 60 / freq_minutes))
    d = df[["pond_id", "timestamp", "y"]].copy(); d["alert"] = (np.asarray(p) >= thr)
    detected, missed, leads = 0, 0, []
    false_alarms = 0; pond_days = 0.0
    for _, g in d.groupby("pond_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        pond_days += len(g) * freq_minutes / 1440
        # an "episode" = consecutive y==1 rows leading to one onset
        y = g["y"].to_numpy(); a = g["alert"].to_numpy()
        i = 0
        while i < len(g):
            if y[i] == 1:
                j = i
                while j < len(g) and y[j] == 1: j += 1
                if a[i:j].any():
                    detected += 1
                    first = int(np.argmax(a[i:j]))
                    leads.append((j - (i + first)) * freq_minutes / 60)
                else:
                    missed += 1
                i = j
            else:
                i += 1
        # false alarms: alert onsets (rising edges) where no y==1 within the same rows
        rising = np.where(a & ~np.concatenate([[False], a[:-1]]))[0]
        for r in rising:
            if not y[r:min(len(g), r + h)].any():
                false_alarms += 1
    tot = detected + missed
    return {"events": tot, "detected": detected,
            "event_recall": detected / tot if tot else float("nan"),
            "false_alarms_per_pond_day": false_alarms / pond_days if pond_days else float("nan"),
            "median_lead_h": float(np.median(leads)) if leads else float("nan")}
