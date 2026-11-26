"""SOP retrieval, single-call generation and deterministic validation (Algorithms 9-11)."""
import json
import re
from typing import Literal
import requests
import yaml
from pydantic import BaseModel, ValidationError


# ---------- SOP retrieval (metadata first) ----------
def load_sops(path):
    with open(path) as f:
        recs = yaml.safe_load(f) or []
    return [r for r in recs if r.get("status") == "approved"]  # drafts never reach the model


def retrieve(sops, species, parameter, trigger=None):
    c = [r for r in sops if species in r["species"] and r["parameter"] == parameter
         and (trigger is None or r.get("trigger") == trigger)]
    return c  # empty list means ABSTAIN


# ---------- contracts ----------
class Action(BaseModel):
    action: str
    sop_id: str


class AdvisoryOut(BaseModel):
    alert_id: str
    risk_summary: str
    explanation: str
    recommended_actions: list[Action]
    uncertainty_note: str
    requires_human_confirmation: Literal[True]


SYSTEM_PROMPT = (
    "You write short advisories for a pond operator. Use ONLY the facts and approved actions in the JSON "
    "input. Never invent numbers, units, actions or sources. Recommend only actions listed in "
    "retrieved_sops and cite their sop_id. If abstain is true or no SOP is given, recommend no actions and "
    "say that no approved procedure is available. State uncertainty. Always set "
    "requires_human_confirmation to true. Do not recommend chemical dosing. Reply with JSON only, with "
    "keys: alert_id, risk_summary, explanation, recommended_actions (list of {action, sop_id}), "
    "uncertainty_note, requires_human_confirmation.")


def build_input(alert_id, pond_id, species, horizon_h, p, risk_class, threshold, latest, trends,
                top_factors, sops, data_status="ok"):
    return {"alert_id": alert_id, "pond_id": pond_id, "species": species, "horizon_h": horizon_h,
            "risk_probability": round(float(p), 3), "risk_class": risk_class,
            "threshold": round(float(threshold), 3), "data_status": data_status,
            "latest": latest, "trends": trends, "top_factors": top_factors,
            "retrieved_sops": [{"sop_id": r["sop_id"], "approved_actions": r["actions"],
                                "contraindications": r.get("contraindications", [])} for r in sops],
            "abstain": len(sops) == 0}


# ---------- one Ollama call ----------
def generate_ollama(inp, model="llama3.1:8b", url="http://localhost:11434", timeout=60):
    """Exactly one call. Returns raw text or None on failure/timeout."""
    try:
        r = requests.post(f"{url}/api/chat", timeout=timeout, json={
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0.2, "num_ctx": 4096},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": json.dumps(inp)}]})
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception:
        return None


# ---------- deterministic validation ----------
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _allowed_numbers(inp):
    vals = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            vals.add(float(o))
    walk(inp)
    return vals


def _numbers_ok(text, allowed, tol=0.051):
    for tok in _NUM.findall(text):
        v = float(tok)
        if not any(abs(v - a) <= tol or abs(v - a * 100) <= 0.51 for a in allowed):
            return False
    return True


def validate(raw, inp):
    """Return (advisory_dict | None, reason). None means use the template."""
    if raw is None:
        return None, "fallback_timeout"
    try:
        adv = AdvisoryOut.model_validate_json(raw)
    except (ValidationError, ValueError):
        return None, "fallback_schema"
    if adv.alert_id != inp["alert_id"]:
        return None, "fallback_alert_id"
    allowed_ids = {s["sop_id"] for s in inp["retrieved_sops"]}
    approved = {(s["sop_id"], a) for s in inp["retrieved_sops"] for a in s["approved_actions"]}
    for a in adv.recommended_actions:
        if a.sop_id not in allowed_ids or (a.sop_id, a.action) not in approved:
            return None, "fallback_unapproved_action"
    if inp["abstain"] and adv.recommended_actions:
        return None, "fallback_actions_when_abstain"
    text = " ".join([adv.risk_summary, adv.explanation, adv.uncertainty_note])
    if not _numbers_ok(text, _allowed_numbers(inp)):
        return None, "fallback_unsupported_number"
    low = text.lower()
    if inp["risk_class"] == "high" and "low risk" in low:
        return None, "fallback_risk_contradiction"
    if not adv.uncertainty_note.strip():
        return None, "fallback_missing_uncertainty"
    return adv.model_dump(), "validated"


def template_advisory(inp):
    acts = [{"action": a, "sop_id": s["sop_id"]} for s in inp["retrieved_sops"] for a in s["approved_actions"]]
    top = ", ".join(f"{f['feature']} ({f['direction']})" for f in inp["top_factors"][:3]) or "not available"
    return {"alert_id": inp["alert_id"],
            "risk_summary": f"Estimated probability of a dissolved oxygen risk event within {inp['horizon_h']} h "
                            f"is {inp['risk_probability']} (class: {inp['risk_class']}).",
            "explanation": f"Main contributing factors: {top}.",
            "recommended_actions": acts if not inp["abstain"] else [],
            "uncertainty_note": "This is a model estimate, not a measurement. An operator must confirm.",
            "requires_human_confirmation": True}


def advise(inp, generator=generate_ollama, **kw):
    """One generator call, then validate, else template. Returns (advisory, status)."""
    raw = generator(inp, **kw)
    adv, status = validate(raw, inp)
    if adv is None:
        return template_advisory(inp), status
    return adv, status
