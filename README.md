# ZeroLeakX — Continuous ATO Detection via Behavioural Biometrics

**PSB Hackathon 2026** · Risk addressed: **Account Takeover (ATO) only**

Banks authenticate users **once** at login and then trust the whole session. In
an Account Takeover the attacker already holds valid credentials (phishing, SIM
swap, breach reuse), so the system sees a "correct password" and lets them in.
ZeroLeakX fixes the real gap: **the absence of continuous identity validation
after login.** It silently watches *how* the person behaves — typing rhythm,
pointer dynamics, navigation, device and transaction context — builds a
behavioural fingerprint of the legitimate user, and detects drift in real time.
When the **Dynamic Trust Score (DTS)** crosses a risk threshold, adaptive
step-up verification fires. No friction for real users; no free pass for
attackers.

This is a **working full-stack implementation**, not a mock — real keystroke
capture, real on-device feature extraction, and a real ML ensemble
(Isolation Forest + XGBoost + temporal drift) making the risk decisions.

---

## Quick start

```bat
:: Windows — one click
run.bat
```

or manually (any OS):

```bash
pip install -r requirements.txt
python server.py
# open http://127.0.0.1:8000
```

Requires Python 3.10+ (developed/tested on 3.14). First run installs
`fastapi, uvicorn, numpy, scikit-learn, xgboost`.

---

## What's in the full build

- **Multi-user + persistent baselines** — log in as any user; their personal
  model is saved to SQLite and reloaded on next login. Enables the **two-person
  live demo**: enrol a baseline, then log in again and let a *different real
  person* type — genuine, non-synthetic detection.
- **Real PyTorch LSTM** — a recurrent autoencoder over the per-keystroke timing
  sequence (the diagram's "LSTM"), measuring drift from each user's *own*
  enrolled rhythm. Trained on CMU sequences (`data/train_lstm.py`).
- **Live SOC console** — admin view of *all* active sessions ranked by risk over
  a **WebSocket** feed, with a persisted, exportable (CSV) audit log.
- **Richer behavioural features** — 10-feature vector (dwell/flight CV,
  long-pause percentile, rhythm entropy), **mouse dynamics** (speed/accel) and
  **paste detection** on sensitive fields.
- **Real geolocation** — the session's home location is resolved from the real
  IP at login (`ip-api.com`).
- **Works on PC *and* mobile** — responsive layout (on phones the SOC monitor
  becomes a slide-up drawer behind a floating button) plus real **touch / swipe
  dynamics** capture feeding the swipe signal where `mousemove` can't fire. Each
  session is tagged `web` vs `mobile` (a `channel` flag) and shown in the SOC
  console.

## How it maps to the architecture diagram

| Diagram stage | In this build |
|---|---|
| User login → Static auth | Session bootstrap (`/api/session/start`) |
| Session established → baseline loaded | `UserModel` per session; population baseline (cold-start) or personal model after Calibrate |
| Behavioural signals / Device / Transaction | Real browser capture: keystroke dynamics, pointer dynamics, navigation; device fingerprint; transaction velocity |
| **Dynamic Trust Score** (LSTM + Isolation Forest + XGBoost) | `backend/engine.py` — IsolationForest + XGBoost + statistical/EWMA temporal drift, blended into the DTS |
| Risk threshold crossed? | Risk bands in `backend/store.py` (`low / moderate / high / critical`) |
| Allow / Step-up / Block + alert SOC | Adaptive action returned per score; OTP step-up; block + incident escalation |
| Continuous loop | Browser streams a feature window every ~1.4 s for the life of the session |

---

## What is real vs. simulated (be honest with judges)

**Real**
- Keystroke dynamics captured from actual `keydown`/`keyup` events: dwell time,
  flight time, typing speed, variability, backspace rate.
- **On-device feature extraction** — raw keystrokes never leave the browser;
  only the numeric feature vector is POSTed (matches the project's
  privacy-by-design / DPDP story).
- Pointer dynamics (movement speed mean/var) → the "swipe" signal on desktop.
- Device fingerprint (canvas + UA + screen + timezone + cores), hashed.
- **The ML is genuine**: `IsolationForest` (unsupervised anomaly),
  `XGBoost` (legit-vs-attacker classifier with native tree-SHAP explanations),
  a statistical Mahalanobis-style distance, and an EWMA temporal-drift signal.
  The Dynamic Trust Score, risk band and step-up decision are all model-driven.
- **Trained against a real dataset**: the attacker/negative class is **real
  impostor keystrokes** from the CMU Keystroke Dynamics benchmark (51 subjects),
  not synthetic data — see *Dataset & validation* below.

**Simulated for the demo**
- The **attacker's input** in the Simulate tab is synthetic (a quick one-click
  demo) — but it is scored by the **real model**, so the detection, DTS, SHAP and
  step-up are authentic. For a **fully real** alternative with zero synthetic
  input, use the two-person login flow (a different human typing against a stored
  baseline) — see the demo script.
- Geo distance is injected as a scenario override (no IP-geolocation service
  wired in). Geo is deliberately a **low-weight contextual** signal, per spec.
- Balances/transactions are demo data; OTP is shown on-screen instead of SMS.

---

## The ML engine (`backend/engine.py`)

Feature vector (10-D), extracted on the backend from per-keystroke timing arrays
the browser sends (single source of truth shared by the dataset, runtime and
live capture — see `engine.extract_features`):

```
mean_dwell, std_dwell, mean_flight, std_flight, typing_speed, backspace_rate,
dwell_cv, flight_cv, flight_p90, rhythm_entropy
```

A real **PyTorch LSTM autoencoder** also scores the raw keystroke *sequence*;
its reconstruction error relative to the user's own enrolled rhythm is the
temporal-drift term (robust to the password-vs-free-text distribution shift).

Pipeline per user:

1. **Enrolment** (Calibrate tab): collect ≥3 windows → fit `StandardScaler`,
   `IsolationForest`, and an `XGBoost` classifier trained on the enrolment
   samples (legit, label 0) vs. **real impostor keystroke vectors** sampled from
   the CMU benchmark pool (attack, label 1). The user's statistical profile
   (per-feature mean/std) is stored. Cold-start before enrolment uses a
   population profile **derived from the real dataset**, with a tightened
   threshold (the spec's cold-start handling).
2. **Scoring** per window:
   - `IsolationForest.decision_function` → anomaly probability
   - `XGBoost.predict_proba` → **ATO probability**
   - statistical distance (mean |z-score|) → behavioural deviation
   - blended `anomaly = 0.25·iso + 0.50·xgb + 0.25·stat`
   - native XGBoost SHAP (`pred_contribs`) → per-feature explanation
3. **Temporal drift** — an EWMA of the per-window anomaly across the session is
   the "LSTM-style" drift verdict (`Normal / Drift detected / Anomaly confirmed`).

### Dynamic Trust Score

Six signals are weighted into the DTS (`backend/store.py`):

```
key 0.34 · swipe 0.16 · nav 0.10 · device 0.18 · geo 0.10 · txn 0.12
```

Keystroke + pointer dominate (hardest to steal with a credential); geo/velocity
are low-weight contextual signals. Risk bands:

| DTS | Risk | Action |
|----|------|--------|
| 80–100 | low | allow, keep monitoring |
| 50–79 | moderate | silent re-check, tighten window |
| 25–49 | high | step-up (OTP / biometric) |
| 0–24 | critical | block, alert SOC, lock |

---

## Dataset & validation

The negative (impostor) class is grounded in a **real, published dataset**:

> **CMU Keystroke Dynamics Benchmark** — Killourhy & Maxion, *"Comparing Anomaly
> Detectors for Keystroke Dynamics"* (DSN 2009). 51 subjects each typing the same
> string 400 times = 20,400 real samples.

[`data/prepare.py`](data/prepare.py) downloads/parses it, converts every typing
sample into our 6-feature vector, and writes three artifacts to
`backend/model_data/`:

- `impostor_pool.npy` — 2,500 **real** other-people keystroke feature vectors,
  used as the attack class when training each user's model at runtime.
- `population.json` — real population mean/std per feature (cold-start baseline).
- `metrics.json` — a **leakage-free** evaluation.

**Reported result** (per-subject genuine vs. impostors from *unseen identities*,
disjoint train/test identities):

| Metric | Value |
|---|---|
| ROC AUC | **0.85** |
| Equal Error Rate | **0.20** |
| Subjects | 51 |
| Features | 10 |

This EER is honest for *this* feature set: ZeroLeakX uses only **session-level
aggregate features** (dwell/flight mean·std·CV·percentile·entropy, speed,
backspace) because those are the ones computable from arbitrary free text.
Per-key digraph models in the literature score lower EER but cannot run on free
typing. The number is shown live in the app (Calibrate tab) and at
`/api/health`. (Re-run `python data/prepare.py` to regenerate.)

> Note: the CMU set is error-free fixed-password typing, so it carries no
> backspace data; `backspace_rate` for impostor vectors is filled with a
> realistic, non-discriminative value so it cannot leak a spurious signal.

To (re)generate the artifacts:

```bash
python data/prepare.py    # writes backend/model_data/*, prints the metrics
```

The engine loads these on startup and falls back to a synthetic impostor
generator only if the artifacts are missing.

## Demo script (≈90 s for judges)

1. **Log in** as `alice` → **Calibrate** → type 4 lines → **Train personal
   model** (saved to SQLite). The DTS gauge + model output (LSTM / Isolation
   Forest / XGBoost / ATO) update live every ~1.4 s.
2. **Two-person test (the real one):** open a second tab, log in as `alice`
   again (loads her baseline), and let a **different person** type in the
   transfer form. Their real rhythm drifts from alice's baseline → DTS drops →
   step-up. No synthetic input anywhere. Watch both sessions in **SOC Console**.
3. **Simulate → Scenario 1 (credential-stuffing takeover)** — DTS collapses to
   **critical** over ~7 s as keystroke, device and geo diverge. Step-up OTP
   fires. Enter a **wrong** code → session terminated, SOC incident raised.
4. **Simulate → Scenario 2 (mid-session behavioural takeover)** — same device,
   different typist → only behaviour drifts → **high** → one-time step-up.
5. **Scenario 4 (new device)** → step-up; enter the **correct** OTP shown in the
   modal → legitimate user resumes seamlessly, device added to trusted list.
6. **Scenario 3 (travel)** → geo collapses but, being low-weight, the clean user
   is only flagged, not challenged — the right call.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/session/start` | New session + initial state |
| POST | `/api/session/{id}/enroll` | Train personal model from feature windows |
| POST | `/api/session/{id}/score` | Score a behaviour window → DTS + breakdown + action |
| POST | `/api/session/{id}/stepup/request` | Issue step-up OTP |
| POST | `/api/session/{id}/stepup/verify` | Verify OTP → resume or block |
| POST | `/api/session/{id}/reset` | Reset session to clean baseline |
| GET | `/api/session/{id}` | Current session state |
| GET | `/api/health` | Engine health + weights |

Interactive docs at `http://127.0.0.1:8000/docs`.

---

## Project layout

```
backend/
  engine.py            ML ensemble: feature extraction, IsolationForest + XGBoost + SHAP
  lstm.py              real PyTorch LSTM autoencoder (temporal drift)
  store.py             session state, DTS aggregation, risk bands, OTP, event log
  db.py                SQLite: persistent user baselines + audit log
  schemas.py           pydantic request models
  app.py               FastAPI routes (login, score, SOC, WebSocket, geo) + static
  model_data/          real training artifacts (built by data/*.py)
    impostor_pool.npy  2,500 real CMU impostor feature vectors (attack class)
    population.json     real cold-start baseline stats
    metrics.json        leakage-free evaluation (AUC / EER)
    lstm.pt, lstm_norm.json   trained LSTM weights + normalisation
data/
  prepare.py           builds model_data/* from the CMU dataset + runs evaluation
  train_lstm.py        trains the LSTM autoencoder on CMU sequences
  DSL-StrongPasswordData.csv   the CMU benchmark (20,400 real samples)
frontend/
  index.html           login + banking UI + SOC dashboard + SOC console + simulator
  app.js               biometric + mouse capture, login, paste, SOC WebSocket, API client
  styles.css           dark SOC theme
server.py              launcher (python server.py)
run.bat                Windows one-click (installs deps + opens browser)
zeroleakx.db           SQLite store (created on first run)
```

---

## Production notes (what scales beyond the hackathon)

> **Full productionisation plan:** see [PRODUCTION.md](PRODUCTION.md) — target
> real-time architecture, latency/SLA budget, fail-open design, the integration
> contracts (capture SDK, bank MFA handoff, SOC/SIEM), the DPDP consent/data
> flow, MLOps, false-positive management, and a phased shadow→adaptive rollout.


- Stateless DTS service → horizontal autoscale (the diagram's GCP Cloud Run).
- Swap the in-memory `SessionStore` for Redis (hot DTS) + Postgres (audit log).
- Federated weekly retraining of personal baselines; confirmed attacks fine-tune
  the XGBoost layer.
- On-device TensorFlow-Lite LSTM for true sequence features; AES-256 encrypted
  feature vectors; ZKP for credential-recovery flows (per the compliance design).
```
