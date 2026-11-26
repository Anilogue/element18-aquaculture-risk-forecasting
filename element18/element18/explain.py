"""SHAP explanations in log-odds space (Algorithm 7)."""
import numpy as np
import shap
import xgboost as xgb


def explain_row(model, x_row, features, top=5):
    """x_row: one-row DataFrame with `features`. Returns dict with base, margin, probability, top factors."""
    expl = shap.TreeExplainer(model)
    phi = np.asarray(expl.shap_values(x_row[features]))[0]
    base = float(np.ravel(expl.expected_value)[0])
    margin = base + float(phi.sum())
    prob = 1.0 / (1.0 + np.exp(-margin))
    model_prob = float(model.predict(xgb.DMatrix(x_row[features]))[0])
    order = np.argsort(-np.abs(phi))[:top]
    return {"base_value": base, "margin": margin, "prob_from_shap": prob, "prob_model": model_prob,
            "consistent": abs(prob - model_prob) < 1e-3,
            "top_factors": [{"feature": features[i], "value": float(x_row[features[i]].iloc[0]),
                             "shap": float(phi[i]), "direction": "up" if phi[i] > 0 else "down"}
                            for i in order]}
