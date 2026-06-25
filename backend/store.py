"""
ZeroLeakX — session state, DTS aggregation and step-up logic.

Holds live per-session state: the per-user behavioural model, the six dashboard
signals, the rolling Dynamic Trust Score, the temporal-drift EWMA, the event
log and the step-up OTP. Durable data (baselines, audit log) lives in db.py.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from . import db
from .engine import UserModel

WEIGHTS = {"key": 0.34, "swipe": 0.16, "nav": 0.10, "device": 0.18,
           "geo": 0.10, "txn": 0.12}
BASELINE = {"key": 92.0, "swipe": 88.0, "nav": 95.0, "device": 100.0,
            "geo": 98.0, "txn": 90.0}


def risk_band(dts: float) -> tuple[str, str]:
    if dts >= 80:
        return "low", "allow"
    if dts >= 50:
        return "moderate", "monitor"
    if dts >= 25:
        return "high", "stepup"
    return "critical", "block"


@dataclass
class Session:
    id: str
    user: str = "guest"
    account: str = "•••• 4821"
    model: UserModel = field(default_factory=UserModel)
    signals: dict = field(default_factory=lambda: dict(BASELINE))
    device_fp: str | None = None
    geo_city: str = "—"
    channel: str = "web"
    dts: float = 94.0
    dts_history: list[float] = field(default_factory=lambda: [94.0] * 30)
    ewma_anom: float = 0.04
    txn_times: list[float] = field(default_factory=list)
    nav_times: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    otp: str | None = None
    blocked: bool = False
    incident: str | None = None
    last_score: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    dts_sum: float = 94.0
    dts_ticks: int = 1

    def log(self, msg: str, level: str = "green") -> None:
        self.events.insert(0, {"t": time.strftime("%H:%M:%S"), "msg": msg, "level": level})
        del self.events[60:]
        try:
            db.log(self.id, self.user, level, msg)
        except Exception:
            pass

    def recompute(self) -> None:
        self.dts = max(0.0, min(100.0, sum(self.signals[k] * w for k, w in WEIGHTS.items())))
        self.dts_history.append(round(self.dts, 1))
        del self.dts_history[: max(0, len(self.dts_history) - 60)]
        r = round(self.dts)
        self.dts_sum += r
        self.dts_ticks += 1
        self.updated = time.time()

    @property
    def avg_dts(self) -> int:
        return round(self.dts_sum / self.dts_ticks)

    def new_otp(self) -> str:
        self.otp = f"{random.randint(0, 999999):06d}"
        return self.otp

    def summary(self) -> dict:
        band, _ = risk_band(self.dts)
        return {"session_id": self.id, "user": self.user, "dts": round(self.dts),
                "risk": band, "blocked": self.blocked, "geo_city": self.geo_city,
                "channel": self.channel, "enrolled": self.model.enrolled,
                "age_s": int(time.time() - self.created),
                "idle_s": int(time.time() - self.updated)}

    def state(self) -> dict:
        band, action = risk_band(self.dts)
        m = self.last_score
        lstm_real = m.get("lstm_anom")
        drift = self.ewma_anom if lstm_real is None else max(self.ewma_anom, lstm_real)
        lstm = ("Normal" if drift < 0.30 else
                "Drift detected" if drift < 0.55 else "Anomaly confirmed")
        return {
            "session_id": self.id,
            "user": self.user,
            "geo_city": self.geo_city,
            "dts": round(self.dts),
            "dts_exact": round(self.dts, 1),
            "avg_dts": self.avg_dts,
            "history": self.dts_history[-30:],
            "signals": {k: round(v) for k, v in self.signals.items()},
            "risk": band,
            "action": action,
            "blocked": self.blocked,
            "enrolled": self.model.enrolled,
            "n_samples": self.model.n_samples,
            "models": {
                "lstm": lstm,
                "lstm_anom": round(drift, 3),
                "lstm_real": None if lstm_real is None else round(lstm_real, 3),
                "iso_score": m.get("iso_score", 0.04),
                "ato_prob": round(m.get("ato_prob", 0.02) * 100, 1),
                "xgb_dts": round(self.dts),
            },
            "contrib": m.get("contrib", {}),
            "events": self.events[:18],
            "incident": self.incident,
        }


class SessionStore:
    def __init__(self) -> None:
        self._s: dict[str, Session] = {}
        self._incident_seq = 0

    def create(self, user: str = "guest") -> Session:
        sid = f"ZLX-{int(time.time()*1000) % 10_000_000:07d}"
        s = Session(id=sid, user=user)
        self._s[sid] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self._s.get(sid)

    def all(self) -> list[Session]:
        # prune very old idle sessions
        now = time.time()
        for k in [k for k, v in self._s.items() if now - v.updated > 1800]:
            del self._s[k]
        return sorted(self._s.values(), key=lambda x: x.dts)

    def next_incident(self) -> str:
        self._incident_seq += 1
        return f"ZLX-2026-0622-{self._incident_seq:04d}"


store = SessionStore()
