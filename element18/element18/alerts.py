"""Event-driven alert state with cooldown (Algorithm 8)."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class AlertState:
    active: bool = False
    cooldown_until: datetime | None = None


@dataclass
class AlertManager:
    threshold: float
    cooldown: timedelta = timedelta(hours=6)
    recover_margin: float = 0.8   # probability must drop below threshold*margin to reset
    states: dict = field(default_factory=dict)

    def step(self, pond_id: str, horizon_h: int, now: datetime, p: float, data_ok: bool = True, ood: bool = False):
        key = (pond_id, horizon_h)
        st = self.states.setdefault(key, AlertState())
        if not data_ok:
            return {"decision": "degraded", "emit": False}
        if p < self.threshold:
            if st.active and p < self.threshold * self.recover_margin:
                st.active = False
            return {"decision": "no_alert", "emit": False}
        if st.active or (st.cooldown_until and now < st.cooldown_until):
            return {"decision": "suppressed", "emit": False}
        st.active = True
        st.cooldown_until = now + self.cooldown
        return {"decision": "alert", "emit": True, "lowered_confidence": ood}
