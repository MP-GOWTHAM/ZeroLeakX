# ZeroLeakX — Productionisation Roadmap

How the hackathon demonstrator becomes a real-time Account-Takeover (ATO)
defence inside a live bank. This document is the engineering + governance plan:
target architecture, the integration contracts, the data/consent flow, the
fail-open/SLA design, and a phased rollout with explicit exit criteria.

> The prototype in this repo proves the **architecture and ML are correct**.
> Going to production is a data-scale, false-positive-tuning, integration and
> compliance program — not a rewrite of the core idea.

---

## 1. Where the prototype stands vs. production

| Area | Prototype (this repo) | Production target |
|---|---|---|
| Training data | CMU benchmark (51 users, fixed password) | Bank's own customers; population cold-start from millions of real sessions |
| Model accuracy | AUC 0.85 / EER 0.20 on 10 aggregate features | < 1–3% false-positive at chosen operating point; multi-signal fusion |
| Capture | Browser JS (keystroke + pointer) | Hardened iOS / Android / Web SDKs; touch, sensor, network signals |
| Serving | single in-memory FastAPI process | Stateless, autoscaled, p99 < 100 ms, millions of concurrent sessions |
| State | in-memory dict + SQLite | Redis (hot DTS) + Kafka (events) + Postgres/warehouse (audit) + feature store |
| Decision | self-contained OTP modal | Calls the bank's real IAM / MFA / step-up orchestrator |
| On failure | blocks in place | **Fail-open**, shadow-first rollout |
| Compliance | README note | DPDP/GDPR consent, DPIA, encryption, retention, model governance, fairness |

---

## 2. Target real-time architecture

Mirrors the deployment diagram. Five stages, all asynchronous to the banking
app so scoring **never** sits on the customer's critical path:

1. **Capture (on-device).** Native + web SDKs collect behavioural signals and
   extract **feature vectors locally**. Raw keystroke content never leaves the
   device — only timing/derived features (privacy-by-design, DPDP-aligned).
2. **Ingest.** API gateway (OAuth2/JWT, mTLS) → event stream (Kafka) → online
   **feature store** (Redis hot cache + offline store for training parity).
3. **Score.** Stateless risk engine loads the user's model from the feature
   store, runs the ensemble (Isolation Forest + XGBoost + LSTM), and emits the
   **Dynamic Trust Score (DTS)** in < 100 ms. Horizontally autoscaled; scales to
   zero off-peak.
4. **Decide.** DTS → risk band → action (see §4). Step-up is delegated to the
   bank's existing MFA; ZeroLeakX never owns the auth factor itself.
5. **Act.** SOC/SIEM stream, case management/fraud rails, audit store, customer
   notification.

Cross-cutting: **MLOps**, **Compliance**, **Resilience** (see §6, §8, §10).

---

## 3. Latency & SLA budget

| Hop | Budget (p99) |
|---|---|
| On-device feature extraction | 10 ms |
| Network + gateway | 20 ms |
| Feature-store fetch (Redis) | 5 ms |
| Ensemble inference (ONNX/Triton) | 30 ms |
| Decision + emit | 10 ms |
| **End-to-end (async, off critical path)** | **< 100 ms** |

Service SLOs: 99.95% availability; scoring is **best-effort** — a missed or slow
score degrades to the last known DTS, never delays the transaction.

---

## 4. Decision policy (risk bands)

Carried over from the prototype's `risk_band()` and tuned per segment in shadow
mode:

| DTS | Band | Action | Customer experience |
|----|------|--------|---------------------|
| 80–100 | Low | Allow, keep monitoring | Zero friction |
| 50–79 | Moderate | Silent re-check, tighten window | None |
| 25–49 | High | **Step-up** via bank MFA | One challenge |
| 0–24 | Critical | Block, lock, alert SOC | Session terminated |

Thresholds are **policy, not hardcoded** — owned by the fraud/risk team, A/B
tested, and adjustable per channel, product, and customer segment.

---

## 5. Integration contracts

### 5.1 Capture SDK → Risk API

`POST /v2/score` (the prototype's `/api/session/{id}/score`, hardened):

```jsonc
{
  "session_id": "…", "user_ref": "hashed-customer-id",
  "channel": "mobile|web",
  "window": { "dwell": [..], "flight": [..], "backspace_rate": 0.04 },
  "pointer": [speed_mean, speed_std, accel_std],
  "device_fp": "…", "ip": "…", "ts": 1718000000
}
```

Response:

```jsonc
{
  "dts": 84, "risk": "low", "action": "allow",
  "models": { "iso": 0.04, "ato_prob": 0.02, "lstm": 0.1 },
  "contrib": { "rhythm_entropy": 31, "...": "…" },   // SHAP, for SOC/audit
  "decision_id": "…"                                 // for the audit trail
}
```

### 5.2 Step-up handoff → bank IAM/MFA

ZeroLeakX **requests** a step-up; the bank's orchestrator owns the factor:

```
POST /iam/step-up { customer_id, reason: "behavioural_anomaly",
                    decision_id, min_assurance: "AAL2" }
→ { challenge_id, status }
ZeroLeakX consumes the IAM webhook → on PASS restores trust, on FAIL escalates.
```

No OTP/biometric secret is ever stored or verified inside ZeroLeakX.

### 5.3 SOC / SIEM

Every decision and band transition is emitted as a CEF/JSON event to the SIEM
(Splunk/QRadar/Sentinel) with the SHAP explanation attached, plus a session
replay reference. Critical events open a case in the existing fraud workflow.

---

## 6. Resilience & fail-open

- **Fail-open is mandatory.** If the engine, feature store, or model is
  unavailable, sessions default to **allow + log**, never block. A risk-engine
  outage must not become a customer-facing outage.
- **Circuit breaker** on the score call with a tight timeout; on trip, the app
  proceeds and the event is queued for offline scoring.
- **Graceful degradation:** missing signals (no mouse on mobile, cold-start
  user) reduce confidence, not availability — the prototype already does this
  (population model + signal recovery).
- **Shadow safety net:** every new model ships in shadow before it can act.

---

## 7. Data & consent flow (DPDP Act 2023 / RBI)

1. **Consent** captured at onboarding/first use (behavioural biometrics = personal
   data); purpose-limited to fraud prevention; withdrawable.
2. **On-device extraction** — raw keystrokes/coordinates never transmitted; only
   feature vectors, encrypted in transit (TLS) and at rest (AES-256).
3. **Data minimisation** — store the DTS, decision, and SHAP, not raw signals.
4. **Retention** — purge per policy (e.g. 90 days for raw features) with a DPIA on
   file; right-to-erasure honoured.
5. **Fairness** — monitor for disparate impact (tremor, disability, age, shared
   or assistive devices); never make behaviour the *sole* basis of an adverse
   action without a human-reviewable path.

---

## 8. Model lifecycle / MLOps

- **Cold-start:** new customers scored by a population model (tightened
  thresholds) for the first 5–7 sessions while a personal baseline forms — as in
  the prototype.
- **Personal baseline:** built from the customer's own genuine sessions; the
  LSTM drift term is measured relative to *their* rhythm.
- **Federated / on-device retraining** where feasible so raw behaviour stays on
  the device; confirmed-fraud labels feed a supervised fine-tune loop.
- **Drift monitoring:** population and per-segment feature drift; auto-flag for
  retrain. Model registry with versioning and one-click rollback.
- **Continuous evaluation:** champion/challenger, replayed against labelled
  traffic; promotion gated on the metrics in §9.

---

## 9. Accuracy targets & evaluation

| Metric | Target | Why |
|---|---|---|
| ATO catch rate (recall) | 80–90% | Headline efficacy |
| **False-positive rate** | **< 1–3%** | The cost driver — see §11 |
| Step-up rate (genuine users) | < 2% | Friction budget |
| Inference latency (p99) | < 100 ms | Real-time SLA |

Evaluation protocol: leakage-free, identity-disjoint train/test (as in
`data/prepare.py`), plus replay on real labelled traffic during shadow mode.
Report AUC, EER, and the full ROC so the operating point is a deliberate choice.

---

## 10. Phased rollout (with exit criteria)

| Phase | What | Action taken | Exit criteria |
|---|---|---|---|
| **0 — Shadow** | Score live traffic, log only | None | 4–8 weeks stable; FP rate measured per segment |
| **1 — Score-only** | DTS visible to SOC analysts | Manual review only | SOC trusts scores; no analyst overload |
| **2 — Step-up (high-risk)** | Auto step-up at DTS 25–49 | One MFA challenge | Step-up rate < 2%; no support spike |
| **3 — Full adaptive** | Block at critical, lock + SOC | Full policy | FP < target; measured ATO reduction; audit signed off |

Each phase is reversible and ramped by customer cohort.

---

## 11. The hardest problem: false positives

At bank volume, a 2% FP rate = millions of legitimate customers challenged or
blocked per day — call-centre load, churn, reputational damage. Mitigations:

- Shadow-first, segment-specific threshold tuning.
- Multi-signal fusion (behaviour is one input, not the verdict).
- Conservative cold-start and explicit fail-open.
- Human-in-the-loop for adverse actions; easy customer recovery path.
- Continuous FP monitoring as a first-class SLO, not an afterthought.

---

## 12. Security of the system itself

- The SDK is an attack surface: anti-tamper, integrity attestation, replay
  protection (nonce + server-side timestamp checks), bot/automation detection.
- Encrypted feature vectors; optional secure enclave (TEE) for server-side ops.
- Rate limiting and abuse controls on the score API.
- Threat-model the pipeline (poisoning of the personal baseline, adversarial
  mimicry) and red-team before each phase.

---

## 13. Build vs. buy

Most banks **buy** this capability (BioCatch, Callsign, LexisNexis/BehavioSec,
Mastercard NuData) because the moat is the **data and tuning**, not the code.
Build in-house only with committed data-science capacity to own model ops
long-term. This repo is a credible reference implementation for either path —
to evaluate vendors against, or to seed an in-house build.

---

## 14. Indicative team & timeline (pilot)

3–6 months to a cohort pilot with a small cross-functional team: ML engineer,
backend, mobile SDK, security, and compliance/DPO. Full rollout is longer and
gated by the §10 exit criteria. The long pole is not the model — it is shadow
tuning, MFA/SIEM integration, and the DPIA/security sign-off.

---

## 15. Open risks

- False-positive cost at scale (mitigated by §10–11).
- Behavioural drift over time (mitigated by retraining, §8).
- Privacy/consent and fairness exposure (mitigated by §7).
- Adversarial mimicry / SDK tampering (mitigated by §12).
- Cold-start coverage for low-frequency users.
