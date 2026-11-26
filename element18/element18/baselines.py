"""Baselines: threshold rule, persistence and logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .split import scoring_rows


def threshold_rule_score(df, warn_mg_l=3.5):
    """Higher score for lower current DO. Returns a score in [0,1]."""
    return np.clip((warn_mg_l - df["do_mg_l"].to_numpy()) / warn_mg_l, 0, 1)


def persistence_score(df, theta=2.0):
    """Risk proportional to recent falling and nearness to theta, using DO rate only."""
    r = -df["do_rate_3"].fillna(0).to_numpy()
    prox = np.clip((df["do_mg_l"].to_numpy() - theta) / 6.0, 0, 1)
    return np.clip(r, 0, None) / (1 + prox * 10)


def fit_logreg(parts, features):
    tr = scoring_rows(parts["train"])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    return m.fit(tr[features], tr["y"])
