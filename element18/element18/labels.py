"""Event and onset-label construction (Algorithm 3).

E(t)=1 if DO < theta for k consecutive samples.
Y(t,H)=1 iff E(t)=0 and an event onset occurs in (t, t+H].
Rows where E(t)=1 are marked active and excluded from early-warning scoring.
"""
import numpy as np
import pandas as pd


def _pond_labels(do: np.ndarray, theta: float, k: int, h_steps: int):
    n = len(do)
    below = (do < theta).astype(np.int8)
    # E[t]=1 iff the last k samples (including t) are all below theta
    cs = np.concatenate([[0], np.cumsum(below)])
    E = np.zeros(n, dtype=np.int8)
    for t in range(k - 1, n):
        if cs[t + 1] - cs[t + 1 - k] == k:
            E[t] = 1
    onset = np.zeros(n, dtype=np.int8)
    onset[0] = E[0]
    onset[1:] = (E[1:] == 1) & (E[:-1] == 0)
    # future onset count in (t, t+H]
    co = np.concatenate([[0], np.cumsum(onset)])
    Y = np.zeros(n, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)  # enough future to judge the label
    for t in range(n):
        end = min(n - 1, t + h_steps)
        Y[t] = 1 if (co[end + 1] - co[t + 1]) > 0 else 0
        valid[t] = (t + h_steps) <= n - 1
    Y[E == 1] = 0
    return E, onset, Y, valid


def add_labels(df: pd.DataFrame, theta=2.0, k=2, horizon_h=12, freq_minutes=60):
    """df must be sorted by pond_id, timestamp with regular sampling."""
    h_steps = int(round(horizon_h * 60 / freq_minutes))
    out = df.copy()
    cols = {"event_active": [], "onset": [], "y": [], "label_valid": []}
    for _, g in out.groupby("pond_id", sort=False):
        E, onset, Y, valid = _pond_labels(g["do_mg_l"].to_numpy(), theta, k, h_steps)
        cols["event_active"].append(E); cols["onset"].append(onset)
        cols["y"].append(Y); cols["label_valid"].append(valid)
    for c, v in cols.items():
        out[c] = np.concatenate(v)
    out["label_valid"] = out["label_valid"].astype(bool)
    return out
