import os, sys, json
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from element18 import data, labels, features, split, metrics, advisory
from element18.alerts import AlertManager
from datetime import datetime, timedelta, timezone


def series(vals, pond="P1"):
    t = pd.date_range("2025-01-01", periods=len(vals), freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": t, "pond_id": pond, "do_mg_l": vals, "temp_c": 28.0, "ph": 7.5})


def test_validate_rejects_bad_rows():
    df = series([5.0, 5.0, 5.0]); df.loc[1, "do_mg_l"] = -1.0
    clean, rej = data.validate(df)
    assert len(rej) == 1 and len(clean) == 2


def test_validate_keeps_extreme_plausible():
    df = series([5.0, 0.1, 5.0]); clean, rej = data.validate(df)
    assert len(rej) == 0 and (clean["do_mg_l"] == 0.1).any()


def test_label_hand_built_dip():
    do = [6.0] * 10 + [1.0, 1.0, 1.0] + [6.0] * 10     # onset at index 11 (k=2)
    df = labels.add_labels(series(do), theta=2.0, k=2, horizon_h=4)
    assert df.loc[11, "onset"] == 1 and df["onset"].sum() == 1
    assert list(df.index[df["y"] == 1]) == [7, 8, 9, 10]   # (t, t+4] contains 11
    assert df.loc[11, "event_active"] == 1 and df.loc[11, "y"] == 0  # ongoing excluded


def test_label_single_dip_below_k_is_not_event():
    do = [6.0] * 10 + [1.0] + [6.0] * 10
    df = labels.add_labels(series(do), theta=2.0, k=2, horizon_h=4)
    assert df["onset"].sum() == 0


def test_features_are_trailing_only():
    df = series(list(np.linspace(5, 6, 40)))
    f1 = features.build_features(df)
    df2 = df.copy(); df2.loc[30:, "do_mg_l"] = 0.5       # change the future
    f2 = features.build_features(df2)
    cols = features.feature_columns(f1)
    pd.testing.assert_frame_equal(f1.loc[:29, cols], f2.loc[:29, cols])


def test_rolling_mean_against_reference():
    rng = np.random.default_rng(0); df = series(list(5 + rng.normal(0, 1, 60)))
    f = features.build_features(df)
    ref = df["do_mg_l"].rolling(6, min_periods=3).mean()
    np.testing.assert_allclose(f["do_mean_6"].dropna(), ref.dropna())


def test_ammonia_fraction_known_value():
    # Hand calculation: T=301.15 K, pKa=0.09018+2727.92/301.15=9.1485, 10^(9.1485-7.5)=44.5, f=1/45.5=0.0220
    f = float(features.un_ionized_fraction(7.5, 28.0)); assert abs(f - 0.0220) < 0.0005


def test_split_has_no_leakage():
    df = data.synthetic(n_ponds=2, days=60)
    df = labels.add_labels(features.build_features(df), 2.0, 2, 12)
    parts = split.chrono_split(df, 12)   # asserts internally
    assert all(len(v) > 0 for v in parts.values())


def test_alert_dedup_and_cooldown():
    m = AlertManager(threshold=0.5, cooldown=timedelta(hours=6)); t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert m.step("P1", 12, t, 0.9)["emit"] is True
    assert m.step("P1", 12, t + timedelta(hours=1), 0.9)["decision"] == "suppressed"
    assert m.step("P1", 12, t + timedelta(hours=2), 0.1)["decision"] == "no_alert"
    assert m.step("P1", 12, t + timedelta(hours=3), 0.9)["decision"] == "suppressed"   # still cooling down
    assert m.step("P1", 12, t + timedelta(hours=8), 0.9)["emit"] is True
    assert m.step("P2", 12, t, 0.9, data_ok=False)["decision"] == "degraded"


SOPS = [{"sop_id": "S1", "species": ["nile_tilapia"], "parameter": "dissolved_oxygen",
         "actions": ["act A"], "contraindications": [], "status": "approved"}]


def mk_input(sops=SOPS):
    return advisory.build_input("a1", "P1", "nile_tilapia", 12, 0.82, "high", 0.5,
                                {"do_mg_l": 3.1}, {"do_rate_3": -0.6}, [], sops)


def good(inp):
    return json.dumps({"alert_id": "a1", "risk_summary": "Probability 0.82 within 12 h.",
                       "explanation": "DO is 3.1 and falling.", "recommended_actions": [{"action": "act A", "sop_id": "S1"}],
                       "uncertainty_note": "Model estimate.", "requires_human_confirmation": True})


def test_retrieval_abstains_for_unknown_species():
    assert advisory.retrieve(SOPS, "carp", "dissolved_oxygen") == []
    assert advisory.build_input("a", "P", "carp", 12, .5, "low", .5, {}, {}, [], [])["abstain"] is True


def test_validation_accepts_good():
    inp = mk_input(); adv, st = advisory.advise(inp, generator=lambda i: good(i)); assert st == "validated"


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: d.update(recommended_actions=[{"action": "invented", "sop_id": "S1"}]), "fallback_unapproved_action"),
    (lambda d: d.update(recommended_actions=[{"action": "act A", "sop_id": "S9"}]), "fallback_unapproved_action"),
    (lambda d: d.update(explanation="DO is 9.9 and falling."), "fallback_unsupported_number"),
    (lambda d: d.update(requires_human_confirmation=False), "fallback_schema"),
    (lambda d: d.update(alert_id="zzz"), "fallback_alert_id"),
    (lambda d: d.update(uncertainty_note=" "), "fallback_missing_uncertainty"),
])
def test_validation_rejects(mutate, expected):
    inp = mk_input(); d = json.loads(good(inp)); mutate(d)
    adv, st = advisory.advise(inp, generator=lambda i: json.dumps(d)); assert st == expected
    assert adv["requires_human_confirmation"] is True   # template returned


def test_timeout_and_garbage_fall_back():
    inp = mk_input()
    assert advisory.advise(inp, generator=lambda i: None)[1] == "fallback_timeout"
    assert advisory.advise(inp, generator=lambda i: "not json")[1] == "fallback_schema"


def test_abstain_blocks_actions():
    inp = mk_input([]); d = json.loads(good(inp))
    assert advisory.advise(inp, generator=lambda i: json.dumps(d))[1] in ("fallback_unapproved_action", "fallback_actions_when_abstain")
    assert advisory.template_advisory(inp)["recommended_actions"] == []


def test_exactly_one_generator_call():
    calls = []
    advisory.advise(mk_input(), generator=lambda i: calls.append(1) or good(i)); assert len(calls) == 1


def test_event_metrics_counts():
    df = series([6.0] * 30); df["y"] = 0; df.loc[10:12, "y"] = 1
    p = np.zeros(30); p[11] = 0.9
    m = metrics.event_metrics(df, p, 0.5, 12)
    assert m["events"] == 1 and m["detected"] == 1 and m["event_recall"] == 1.0
    p2 = np.zeros(30); m2 = metrics.event_metrics(df, p2, 0.5, 12); assert m2["event_recall"] == 0.0
