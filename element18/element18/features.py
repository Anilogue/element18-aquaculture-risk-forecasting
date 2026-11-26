"""Trailing-window features (Algorithm 2).

`build_core_features` runs on pandas or cuDF; extras are pandas-only.
All windows end at t, so no future information enters a feature.
"""
import numpy as np
import pandas as pd

LAGS = (1, 3, 6, 12)
RATE_K = (1, 3, 6)
WINDOWS = (6, 12, 24)


def build_core_features(df, max_gap=3):
    """Lags, rates, rolling stats and time encoding.

    Uses only operations available in both pandas and cuDF, so the same function
    can be timed on either library. Pass a pandas or cudf DataFrame.
    """
    df = df.copy()
    cols = ["do_mg_l", "temp_c", "ph"]
    for c in cols:
        df[c + "_missing"] = df[c].isna().astype("int8")
    df[cols] = df.groupby("pond_id")[cols].ffill(limit=max_gap)
    grp = df.groupby("pond_id")["do_mg_l"]
    for k in LAGS:
        df[f"do_lag_{k}"] = grp.shift(k)
    for k in RATE_K:
        df[f"do_rate_{k}"] = (df["do_mg_l"] - grp.shift(k)) / k
    for w in WINDOWS:
        r = grp.rolling(w, min_periods=max(2, w // 2))
        for name in ("mean", "std", "min", "max"):
            df[f"do_{name}_{w}"] = getattr(r, name)().reset_index(level=0, drop=True)
    for c in ("temp_c", "ph"):
        gg = df.groupby("pond_id")[c]
        df[f"{c}_rate_3"] = (df[c] - gg.shift(3)) / 3
    hr = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    return df


def add_extra_features(df, ewm_alpha=0.3):
    """pandas-only extras: EWMA and 6-sample least-squares slope of DO."""
    df = df.copy()
    grp = df.groupby("pond_id")["do_mg_l"]
    df["do_ewm"] = grp.transform(lambda s: s.ewm(alpha=ewm_alpha, adjust=False).mean())
    x = np.arange(6) - 2.5
    denom = float((x ** 2).sum())
    def slope(s):
        return s.rolling(6, min_periods=6).apply(
            lambda a: float((x * (a - a.mean())).sum() / denom), raw=True)
    df["do_slope_6"] = grp.transform(slope)
    return df


def build_features(df, max_gap=3, ewm_alpha=0.3):
    """Full pandas feature set used for modelling."""
    return add_extra_features(build_core_features(df, max_gap), ewm_alpha)


FEATURE_PREFIXES = ("do_lag_", "do_rate_", "do_mean_", "do_std_", "do_min_", "do_max_",
                    "do_ewm", "do_slope", "temp_c_rate", "ph_rate", "hour_")
BASE = ["do_mg_l", "temp_c", "ph"]


def feature_columns(df, current_only=False):
    if current_only:
        return list(BASE)
    cols = BASE + [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    return list(dict.fromkeys(cols))


def un_ionized_fraction(ph, temp_c):
    """Fraction of total ammonia nitrogen present as NH3 (freshwater, 0-50 C)."""
    T = np.asarray(temp_c) + 273.15
    pka = 0.09018 + 2727.92 / T
    return 1.0 / (10.0 ** (pka - np.asarray(ph)) + 1.0)
