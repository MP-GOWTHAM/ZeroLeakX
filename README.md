# ZeroLeakX — Continuous ATO Detection via Behavioural Biometrics

Banks authenticate users **once** at login and then trust the whole session. In
an Account Takeover (ATO) the attacker already holds valid credentials —
stolen via phishing, SIM swap or data breach — so the system sees a "correct
password" and lets them in. ZeroLeakX fixes the real gap: **the absence of
continuous identity validation after login.**

It silently monitors *how* the person behaves — typing rhythm, pointer/touch
dynamics, navigation patterns, device fingerprint and transaction context —
builds a behavioural fingerprint of the legitimate user, and detects drift in
real time. When the **Dynamic Trust Score (DTS)** crosses a risk threshold,
adaptive step-up verification fires instantly. No friction for real users; no
free pass for attackers.

**[Live demo →](https://mp-gowtham.github.io/ZeroLeakX/)**

---

## Quick start

```bat
:: Windows — one click
run.bat
```

```bash
# Any OS
pip install -r requirements.txt
python server.py
# open http://127.0.0.1:8000
```

Requires Python 3.10+. Dependencies: FastAPI, uvicorn, numpy, scikit-learn,
xgboost, PyTorch.

---

## Features

- **Real keystroke dynamics** — captured from live `keydown`/`keyup` events
  (dwell time, flight time, typing speed, variability, backspace rate). Raw
  keystrokes never leave the browser — only feature vectors are sent
  (privacy-by-design).
- **Real ML ensemble** — `IsolationForest` (unsupervised anomaly) + `XGBoost`
  (legit-vs-impostor classifier with native tree-SHAP explanations) + a real
  **PyTorch LSTM autoencoder** for temporal drift detection + statistical
  distance. All model-driven, no hardcoded thresholds.
- **Trained on real data** — the impostor/negative class uses **real human
  keystrokes** from the CMU Keystroke Dynamics benchmark (51 subjects, 20,400
  samples). AUC **0.85**, EER **0.20** on unseen identities (leakage-free
  evaluation).
- **10-feature vector** — dwell/flight mean, std, coefficient of variation,
  90th-percentile flight, rhythm entropy, typing speed, backspace rate.
  Extracted backend-side for train/serve parity.
- **Multi-user with persistent baselines** — each user's personal model is
  saved to SQLite and reloaded on login. Supports the two-person test: enrol a
  baseline, then let a different person type → genuine detection.
- **Live SOC console** — admin view of all active sessions ranked by risk,
  streamed via **WebSocket**, with a persisted and exportable (CSV) audit log.
- **Mouse + touch dynamics** — pointer speed/acceleration on desktop,
  `touchmove` swipe velocity on mobile, feeding the swipe signal on both
  platforms.
- **Paste detection** — clipboard paste into sensitive fields (account number,
  amount) is captured and logged as a behavioural signal.
- **Real IP geolocation** — the session's location is resolved from the real IP
  at login.
- **Responsive PC + mobile** — on phones the sidebar collapses, the SOC monitor
  becomes a slide-up drawer behind a floating button, and touch capture replaces
  mouse capture. Sessions are tagged `web` vs `mobile`.
- **Adaptive step-up** — OTP challenge triggered proportionally to risk; wrong
  OTP → session terminated + SOC incident. Correct OTP → session resumes with
  zero friction.

---

## Architecture

```
Browser (on-device)          FastAPI backend             SOC / audit
┌─────────────────┐    ┌───────────────────────┐    ┌──────────────┐
│ Keystroke timing │───→│ Feature extraction    │    │ WebSocket    │
│ Pointer/touch    │    │ (10-D vector)         │    │ live feed    │
│ Device FP        │    │                       │    │              │
│ Paste detection  │    │ Isolation Forest      │    │ Session table│
└─────────────────┘    │ XGBoost + SHAP        │───→│ Audit log    │
                       │ LSTM autoencoder      │    │ CSV export   │
                       │                       │    └──────────────┘
                       │ DTS → risk band       │
                       │ → allow / monitor /   │
                       │   step-up / block     │
                       └───────────────────────┘
```

### Dynamic Trust Score

Six signals weighted into the DTS:

| Signal | Weight | Source |
|--------|--------|--------|
| Keystroke dynamics | 0.34 | Real typing rhythm vs enrolled baseline |
| Device fingerprint | 0.18 | Canvas + UA + screen + timezone hash |
| Swipe / pointer | 0.16 | Mouse or touch movement dynamics |
| Transaction velocity | 0.12 | Burst detection over a 60s window |
| Geo context | 0.10 | IP-based location shift |
| Navigation flow | 0.10 | Page-transition speed patterns |

### Risk bands

| DTS | Risk | Action |
|-----|------|--------|
| 80–100 | Low | Allow, keep monitoring |
| 50–79 | Moderate | Silent re-check, tighten monitoring window |
| 25–49 | High | Step-up authentication (OTP / biometric) |
| 0–24 | Critical | Block session, alert SOC, lock account |

---

## ML pipeline

### Feature extraction

```
mean_dwell, std_dwell, mean_flight, std_flight, typing_speed, backspace_rate,
dwell_cv, flight_cv, flight_p90, rhythm_entropy
```

Single source of truth in `backend/engine.py` — the same function is used by
the CMU dataset preparation, model training and live scoring.

### Enrolment

1. User types ≥3 calibration sentences in the browser.
2. Per-keystroke timing arrays (no key content) are sent to the backend.
3. `StandardScaler` + `IsolationForest` + `XGBoost` are fit on the enrolment
   windows (legit) vs. real impostor keystroke vectors sampled from the CMU pool
   (attack). The user's statistical profile and LSTM reconstruction baseline are
   stored.

### Scoring (per window, every ~1.4s)

- `IsolationForest.decision_function` → anomaly probability
- `XGBoost.predict_proba` → ATO probability
- Statistical distance (mean |z-score| on core features)
- LSTM reconstruction error relative to the user's own enrolled rhythm
- Blended into a single anomaly score → key signal → DTS

### Cold start

Before enrolment, a population-level model (derived from the real CMU dataset)
is used with tightened thresholds. The personal model takes over after
calibration.

---

## Dataset & validation

Trained and evaluated on the **CMU Keystroke Dynamics Benchmark** (Killourhy &
Maxion, DSN 2009) — 51 subjects × 400 repetitions = 20,400 real samples.

| Metric | Value |
|--------|-------|
| ROC AUC | **0.85** |
| Equal Error Rate | **0.20** |
| Subjects | 51 |
| Features | 10 |
| Protocol | Per-subject genuine vs impostor from *unseen* identities, disjoint train/test |

The LSTM autoencoder is separately trained on the same CMU sequences
(`data/train_lstm.py`) — genuine reconstruction error 0.09 vs impostor 0.23.

Regenerate all artifacts:

```bash
python data/prepare.py      # impostor pool + population stats + XGBoost eval
python data/train_lstm.py   # LSTM weights
```

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/login` | Login with username, load/create baseline |
| POST | `/api/session/{id}/enroll` | Train personal model from typing windows |
| POST | `/api/session/{id}/score` | Score a behaviour window → DTS + action |
| POST | `/api/session/{id}/stepup/request` | Issue step-up OTP |
| POST | `/api/session/{id}/stepup/verify` | Verify OTP → resume or block |
| POST | `/api/session/{id}/reset` | Reset session state |
| GET | `/api/session/{id}` | Current session state |
| GET | `/api/health` | Engine status, model info, benchmark metrics |
| GET | `/api/soc/sessions` | All active sessions (SOC view) |
| GET | `/api/soc/audit` | Persisted audit log |
| GET | `/api/soc/export` | Download audit as CSV |
| WS | `/ws/soc` | Live session feed (WebSocket) |

Interactive docs at `http://127.0.0.1:8000/docs`.

---

## Project layout

```
backend/
  engine.py            ML ensemble: feature extraction, IsolationForest + XGBoost + SHAP
  lstm.py              PyTorch LSTM autoencoder (temporal drift)
  store.py             Session state, DTS aggregation, risk bands, OTP
  db.py                SQLite: persistent user baselines + audit log
  schemas.py           Pydantic request models
  app.py               FastAPI routes, WebSocket, geolocation, static serving
  model_data/
    impostor_pool.npy  2,500 real CMU impostor feature vectors
    population.json    Population baseline stats
    metrics.json       Evaluation results (AUC / EER)
    lstm.pt            Trained LSTM weights
    lstm_norm.json     LSTM normalisation parameters
data/
  prepare.py           Build model artifacts from the CMU dataset
  train_lstm.py        Train the LSTM autoencoder
  DSL-StrongPasswordData.csv   CMU benchmark (20,400 samples)
frontend/
  index.html           Banking UI + SOC dashboard + SOC console + simulator
  app.js               Biometric capture, login, paste, WebSocket, API client
  styles.css           Dark SOC theme
docs/
  index.html           Self-contained GitHub Pages demo (no backend)
server.py              Launcher
run.bat                Windows one-click start
```

---

## What is real vs. simulated

| Component | Status |
|-----------|--------|
| Keystroke capture (dwell, flight, speed, backspace) | Real — from live key events |
| On-device feature extraction (raw keys never leave browser) | Real |
| Mouse / touch pointer dynamics | Real |
| Device fingerprint | Real |
| Paste detection on sensitive fields | Real |
| IP geolocation | Real |
| ML models (IsolationForest + XGBoost + LSTM) | Real — genuinely trained and fitted |
| Impostor training data | Real — CMU benchmark, not synthetic |
| SHAP explanations | Real — native XGBoost tree SHAP |
| DTS, risk bands, step-up decisions | Real — model-driven |
| Attacker input in Simulate tab | Synthetic timing (scored by the real model) |
| Two-person login test | Fully real (no synthetic input) |
| Bank balances / transactions | Demo data |
| OTP delivery | Shown on-screen (would be SMS in production) |

---

## Production roadmap

See [PRODUCTION.md](PRODUCTION.md) for the full plan: target real-time
architecture, latency/SLA budget, fail-open design, integration contracts
(capture SDK → bank MFA → SOC/SIEM), DPDP/RBI compliance, MLOps, false-positive
management, and a phased shadow → adaptive rollout.
