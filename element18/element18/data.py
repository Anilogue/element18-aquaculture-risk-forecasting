"""Data loading, validation and a synthetic generator.

The synthetic generator exists only to test code paths and to create scaling
workloads. Models trained on it say nothing about real ponds.
"""
import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "pond_id", "do_mg_l", "temp_c", "ph"]
RANGES = {"do_mg_l": (0.0, 25.0), "temp_c": (0.0, 45.0), "ph": (0.0, 14.0)}


def validate(df: pd.DataFrame, freq_minutes: int = 60):
    """Return (clean_df, rejects_df). Extreme but plausible values are kept."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    bad = df["timestamp"].isna() | df["pond_id"].isna()
    for col, (lo, hi) in RANGES.items():
        v = pd.to_numeric(df[col], errors="coerce")
        df[col] = v
        bad |= v.notna() & ((v < lo) | (v > hi))  # physically impossible
    rejects = df[bad].copy()
    clean = df[~bad].sort_values(["pond_id", "timestamp"])
    clean = clean.drop_duplicates(["pond_id", "timestamp"]).reset_index(drop=True)
    return clean, rejects


def load_csv(path: str, freq_minutes: int = 60):
    return validate(pd.read_csv(path), freq_minutes)


def synthetic(n_ponds=3, days=120, freq_minutes=60, seed=42) -> pd.DataFrame:
    """Diel DO cycle plus slow weather-like stress and occasional crashes."""
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * 60 / freq_minutes)
    t = pd.date_range("2025-01-01", periods=steps, freq=f"{freq_minutes}min", tz="UTC")
    hours = (t.hour + t.minute / 60).values
    frames = []
    for p in range(n_ponds):
        base = 6.0 + rng.normal(0, 0.3)
        diel = 2.2 * np.sin(2 * np.pi * (hours - 9) / 24)
        # AR(1) slow component (cloudy spells lower DO)
        slow = np.zeros(steps)
        for i in range(1, steps):
            slow[i] = 0.995 * slow[i - 1] + rng.normal(0, 0.08)
        do = base + diel + slow + rng.normal(0, 0.15, steps)
        # crash events: fast fall then partial recovery
        n_ev = max(4, days // 8)
        starts = np.linspace(72, steps - 72, n_ev).astype(int) + rng.integers(-24, 24, n_ev)
        for s in starts:
            L = rng.integers(6, 14)
            shape = np.concatenate([np.linspace(0, -4.5, L // 2), np.linspace(-4.5, 0, L - L // 2)])
            do[s:s + L] += shape
        do = np.clip(do, 0.05, None)
        temp = 28 + 2.5 * np.sin(2 * np.pi * (hours - 15) / 24) + rng.normal(0, 0.2, steps)
        ph = 7.6 + 0.5 * np.sin(2 * np.pi * (hours - 14) / 24) + rng.normal(0, 0.05, steps)
        frames.append(pd.DataFrame({"timestamp": t, "pond_id": f"P{p+1}",
                                    "do_mg_l": do, "temp_c": temp, "ph": ph}))
    return pd.concat(frames, ignore_index=True)
