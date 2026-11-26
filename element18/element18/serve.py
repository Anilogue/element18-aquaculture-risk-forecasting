"""FastAPI orchestration: /predict returns the alert at once, advisory is attached if ready.

Run: uvicorn element18.serve:app --port 8000
The alert never waits for the language model beyond a bounded timeout.
"""
import os, uuid
from datetime import datetime, timezone
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from .advisory import load_sops, retrieve, build_input, advise
from .alerts import AlertManager
from .explain import explain_row

MODEL_PATH = os.environ.get("E18_MODEL", "results/exp_001/model.json")
THRESHOLD = float(os.environ.get("E18_THRESHOLD", "0.5"))
SOP_PATH = os.environ.get("E18_SOPS", "knowledge_base/sops_example.yaml")
app = FastAPI(title="Element 18")
manager = AlertManager(threshold=THRESHOLD)
_state = {}


class Row(BaseModel):
    pond_id: str
    horizon_h: int = 12
    features: dict[str, float]
    use_llm: bool = True


def _model():
    if "m" not in _state:
        b = xgb.Booster(); b.load_model(MODEL_PATH)
        _state["m"] = b; _state["feats"] = b.feature_names
    return _state["m"], _state["feats"]


@app.get("/health")
def health():
    return {"status": "ok", "model_present": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict(r: Row):
    m, feats = _model()
    x = pd.DataFrame([{f: r.features.get(f) for f in feats}])
    p = float(m.predict(xgb.DMatrix(x))[0])
    now = datetime.now(timezone.utc)
    dec = manager.step(r.pond_id, r.horizon_h, now, p)
    resp = {"pond_id": r.pond_id, "probability": p, "decision": dec["decision"], "advisory": None}
    if dec["emit"] and r.use_llm:
        ex = explain_row(m, x, feats)
        sops = retrieve(load_sops(SOP_PATH), "nile_tilapia", "dissolved_oxygen")
        inp = build_input(str(uuid.uuid4()), r.pond_id, "nile_tilapia", r.horizon_h, p,
                          "high" if p >= 0.8 else "medium", THRESHOLD,
                          {"do_mg_l": float(x.get("do_mg_l", [float("nan")])[0])}, {},
                          ex["top_factors"], sops)
        adv, status = advise(inp)
        resp["advisory"] = {"content": adv, "validation": status}
    return resp
