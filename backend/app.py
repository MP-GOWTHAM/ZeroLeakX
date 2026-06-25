"""
ZeroLeakX — FastAPI application.

Continuous, post-login identity validation for online banking. Behavioural-
biometric timing windows are scored by a real ML ensemble (Isolation Forest +
XGBoost + LSTM + temporal drift) into a live Dynamic Trust Score; adaptive
step-up fires when risk crosses the threshold. Multi-user with persistent
baselines, a live SOC console (WebSocket), real IP geolocation and an audit log.
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .engine import extract_features, vector
from .schemas import EnrollReq, LoginReq, ScoreReq, StepupVerifyReq
from .store import BASELINE, WEIGHTS, risk_band, store

app = FastAPI(title="ZeroLeakX", version="2.0",
              description="Continuous ATO detection via behavioural biometrics")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
SENSITIVE_FIELDS = {"bene-acct", "txn-amt", "bene-ifsc"}
db.conn()  # initialise database


@app.middleware("http")
async def no_cache_assets(request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith((".js", ".css", ".html")):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


# ── Real IP geolocation (cached once per process) ────────────────────────────
_GEO: dict | None = None


def get_geo() -> dict:
    global _GEO
    if _GEO is not None:
        return _GEO
    try:
        url = "http://ip-api.com/json/?fields=status,city,regionName,country,lat,lon"
        with urllib.request.urlopen(url, timeout=3) as r:
            d = json.loads(r.read().decode())
        if d.get("status") == "success":
            _GEO = {"city": f"{d['city']}, {d['country']}", "lat": d["lat"], "lon": d["lon"]}
        else:
            _GEO = {"city": "Unknown location", "lat": None, "lon": None}
    except Exception:
        _GEO = {"city": "Localhost network", "lat": None, "lon": None}
    return _GEO


# ── Signal helpers ──────────────────────────────────────────────────────────
def _recover(s, key: str, rate: float = 0.18) -> None:
    s.signals[key] += (BASELINE[key] - s.signals[key]) * rate


def _geo_score(km: float) -> float:
    return max(3.0, min(98.0, 98.0 - km / 4.0))


def _nav_score(s) -> float:
    now = time.time()
    s.nav_times = [t for t in s.nav_times if now - t < 8.0]
    return max(40.0, min(98.0, 98.0 - max(0, len(s.nav_times) - 3) * 15.0))


def _txn_score(s) -> float:
    now = time.time()
    s.txn_times = [t for t in s.txn_times if now - t < 60.0]
    return max(20.0, min(95.0, 95.0 - max(0, len(s.txn_times) - 1) * 22.0))


def _load_user_model(s, username: str) -> dict:
    u = db.get_user(username)
    if u and len(u.get("baseline") or []) >= 3:
        s.model.enroll(u["baseline"], u.get("pointer"))
        s.signals.update(BASELINE)
        return {"enrolled": True, "samples": u["n_samples"]}
    return {"enrolled": False, "samples": 0}


# ── Auth / session ──────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    from .engine import _HAS_XGB, DATA_BACKED, METRICS, IMPOSTOR_POOL, _LSTM
    return {"ok": True, "xgboost": _HAS_XGB, "lstm": _LSTM is not None,
            "weights": WEIGHTS, "data_backed": DATA_BACKED,
            "impostor_pool_size": 0 if IMPOSTOR_POOL is None else int(len(IMPOSTOR_POOL)),
            "metrics": METRICS}


@app.get("/api/users")
def users() -> dict:
    return {"users": db.list_users()}


@app.post("/api/login")
def login(req: LoginReq) -> dict:
    username = (req.username or "guest").strip()[:24] or "guest"
    db.upsert_user(username)
    s = store.create(user=username)
    s.geo_city = get_geo()["city"]
    loaded = _load_user_model(s, username)
    s.recompute()
    if loaded["enrolled"]:
        s.log(f"{username} authenticated — personal model loaded ({loaded['samples']} samples)", "green")
    else:
        s.log(f"{username} authenticated — no baseline yet, population model active", "amber")
    out = s.state()
    out["enroll_hint"] = not loaded["enrolled"]
    out["geo_home"] = s.geo_city
    return out


@app.post("/api/session/start")
def session_start() -> dict:
    s = store.create(user="guest")
    s.geo_city = get_geo()["city"]
    s.recompute()
    s.log("Guest session — behavioural baseline loading", "green")
    return s.state()


@app.post("/api/session/{sid}/enroll")
def enroll(sid: str, req: EnrollReq) -> dict:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    raws = [{"dwell": w.dwell, "flight": w.flight, "backspace_rate": w.backspace_rate}
            for w in req.windows]
    res = s.model.enroll(raws, req.pointer)
    if req.device_fp:
        s.device_fp = req.device_fp
    if res.get("enrolled"):
        db.save_baseline(s.user, raws, req.pointer, res["samples"])
        s.signals.update(BASELINE)
        s.ewma_anom = 0.04
        s.recompute()
        s.log(f"Baseline enrolled for {s.user} — {res['samples']} samples, personal model active & saved", "blue")
    else:
        s.log(f"Enrolment incomplete ({res.get('samples', 0)} samples)", "amber")
    out = s.state()
    out["enroll"] = res
    return out


@app.post("/api/session/{sid}/score")
def score(sid: str, req: ScoreReq) -> dict:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    if s.blocked:
        return s.state()

    if req.channel:
        s.channel = req.channel
    prev_band, _ = risk_band(s.dts)
    touched: set[str] = set()

    if req.device_ok is not None:
        s.signals["device"] = 100.0 if req.device_ok else 12.0
        touched.add("device")
        if not req.device_ok:
            s.log("New device detected — fingerprint hash mismatch", "amber")
    elif req.device_fp:
        if s.device_fp is None:
            s.device_fp = req.device_fp
        s.signals["device"] = 100.0 if req.device_fp == s.device_fp else 12.0
        touched.add("device")

    if req.window is not None and (req.window.dwell or req.window.flight):
        res = s.model.score(req.window.dwell, req.window.flight, req.window.backspace_rate)
        s.signals["key"] = res["key_score"]
        s.last_score = res
        la = res.get("lstm_anom")
        s.ewma_anom = 0.6 * s.ewma_anom + 0.4 * (la if la is not None else res["anomaly"])
        touched.add("key")
        if req.pointer:
            sw = s.model.score_pointer(req.pointer)
            if sw is not None:
                s.signals["swipe"] = sw
                touched.add("swipe")

    if req.swipe is not None:
        s.signals["swipe"] = max(0.0, min(100.0, req.swipe)); touched.add("swipe")
    if req.nav_score is not None:
        s.signals["nav"] = max(0.0, min(100.0, req.nav_score)); touched.add("nav")

    if req.geo_km is not None:
        s.signals["geo"] = _geo_score(req.geo_km); touched.add("geo")
        if req.geo_km > 50:
            s.log(f"Geo anomaly: login {int(req.geo_km)} km from {s.geo_city}", "amber")

    if req.nav:
        s.nav_times.append(time.time()); s.signals["nav"] = _nav_score(s); touched.add("nav")

    if req.txn:
        s.txn_times.append(time.time()); s.signals["txn"] = _txn_score(s); touched.add("txn")

    if req.paste:
        lvl = "amber" if req.paste in SENSITIVE_FIELDS else "blue"
        s.log(f"Clipboard paste into '{req.paste}' captured", lvl)

    if not touched or req.label == "live":
        for k in BASELINE:
            if k not in touched:
                _recover(s, k)
        s.ewma_anom = s.ewma_anom * 0.85 + 0.04 * 0.15

    s.recompute()

    band, action = risk_band(s.dts)
    if band != prev_band:
        if band == "critical":
            s.log("CRITICAL — DTS below threshold. Step-up authentication required", "red")
        elif band == "high":
            s.log(f"High risk — DTS {round(s.dts)}. Step-up challenge triggered", "amber")
        elif band == "moderate":
            s.log(f"Moderate risk — DTS {round(s.dts)}. Monitoring tightened", "amber")
        elif band == "low" and prev_band in ("moderate", "high", "critical"):
            s.log("Session trust restored — DTS back in safe range", "green")
    return s.state()


@app.post("/api/session/{sid}/stepup/request")
def stepup_request(sid: str) -> dict:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    code = s.new_otp()
    s.log("Step-up authentication triggered — OTP sent to •••• 7291", "amber")
    return {"phone": "•••• 7291", "demo_code": code, "message": "OTP dispatched"}


@app.post("/api/session/{sid}/stepup/verify")
def stepup_verify(sid: str, req: StepupVerifyReq) -> dict:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    if req.code == s.otp:
        s.signals.update(BASELINE)
        s.ewma_anom = 0.04
        s.blocked = False
        s.last_score = {}
        s.recompute()
        s.log("Step-up PASSED — user identity confirmed, session resumed", "green")
        out = s.state(); out["verified"] = True
        return out
    s.blocked = True
    s.incident = store.next_incident()
    for k in s.signals:
        s.signals[k] = max(6.0, s.signals[k] * 0.1)
    s.recompute()
    s.log("OTP FAILED — attacker confirmed, session forcibly terminated", "red")
    s.log("Account temporarily locked — user notified via SMS & email", "red")
    s.log(f"SOC escalation — Incident #{s.incident}", "red")
    out = s.state(); out["verified"] = False
    return out


@app.post("/api/session/{sid}/reset")
def reset(sid: str) -> dict:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    s.signals.update(BASELINE)
    s.ewma_anom = 0.04
    s.blocked = False
    s.incident = None
    s.last_score = {}
    s.txn_times.clear()
    s.nav_times.clear()
    s.recompute()
    s.log("Session reset — behavioural baseline reloaded", "green")
    return s.state()


@app.get("/api/session/{sid}")
def get_session(sid: str) -> dict:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return s.state()


# ── SOC console ──────────────────────────────────────────────────────────────
@app.get("/api/soc/sessions")
def soc_sessions() -> dict:
    return {"sessions": [s.summary() for s in store.all()], "audit": db.audit(30)}


@app.get("/api/soc/audit")
def soc_audit(limit: int = 100) -> dict:
    return {"audit": db.audit(limit)}


@app.get("/api/soc/export")
def soc_export() -> PlainTextResponse:
    rows = db.audit(2000)
    lines = ["time,session_id,user,level,message"]
    for r in rows:
        msg = r["msg"].replace('"', "'")
        lines.append(f'{r["t"]},{r["session_id"]},{r["username"]},{r["level"]},"{msg}"')
    return PlainTextResponse("\n".join(lines), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=zeroleakx_audit.csv"})


@app.websocket("/ws/soc")
async def ws_soc(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({"sessions": [s.summary() for s in store.all()],
                                "audit": db.audit(20)})
            await asyncio.sleep(1.0)
    except Exception:
        return


# ── Static frontend ─────────────────────────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
