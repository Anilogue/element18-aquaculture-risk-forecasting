"""Chronological split with purge gaps (Algorithm 4)."""
import pandas as pd


def chrono_split(df, horizon_h=12, freq_minutes=60, frac=(0.6, 0.15, 0.10, 0.15)):
    assert abs(sum(frac) - 1) < 1e-9
    gap = pd.Timedelta(hours=horizon_h)
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    span = t1 - t0
    c = [t0 + span * sum(frac[:i + 1]) for i in range(3)]
    ts = df["timestamp"]
    parts = {
        "train": df[ts < c[0] - gap],
        "valid": df[(ts >= c[0]) & (ts < c[1] - gap)],
        "calib": df[(ts >= c[1]) & (ts < c[2] - gap)],
        "test": df[ts >= c[2]],
    }
    order = ["train", "valid", "calib", "test"]
    for a, b in zip(order, order[1:]):
        if len(parts[a]) and len(parts[b]):
            assert parts[a]["timestamp"].max() + gap <= parts[b]["timestamp"].min() + pd.Timedelta(0), \
                f"leakage: {a} label window reaches {b}"
    return parts


def scoring_rows(part):
    """Rows usable for early-warning scoring: not in an event, label fully observed."""
    return part[(part["event_active"] == 0) & (part["label_valid"])].dropna()
