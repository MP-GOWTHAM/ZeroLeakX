"""
ZeroLeakX — behavioural-biometrics risk engine.

Realises the "LSTM + Isolation Forest + XGBoost" stack from the architecture
diagram with genuine, runnable ML:

  * Isolation Forest  -> unsupervised anomaly detection on the keystroke feature
                         vector.
  * XGBoost           -> supervised legit-vs-impostor classifier (ATO
                         probability) trained per user from the enrolment
                         baseline plus REAL impostor keystrokes (CMU benchmark).
                         Native tree-SHAP explains every decision.
  * LSTM              -> a real PyTorch recurrent autoencoder over the
                         per-keystroke timing sequence; reconstruction error is
                         the temporal behavioural-drift signal (see lstm.py).
  * Statistical / EWMA-> Mahalanobis-style distance + session drift.

Feature extraction lives here (single source of truth) so the CMU dataset prep,
the live runtime, and scoring all compute identical vectors. The browser sends
per-keystroke *timing* arrays (no key content) for each window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from . import lstm as lstm_mod
    _LSTM = lstm_mod.load_detector()
except Exception:
    _LSTM = None

# ── Feature contract (10-D) ──────────────────────────────────────────────────
FEATURES = [
    "mean_dwell",      # avg key hold time (ms)
    "std_dwell",       # variability of hold time
    "mean_flight",     # avg gap between keys (ms)
    "std_flight",      # variability of that gap
    "typing_speed",    # keys per second
    "backspace_rate",  # share of keys that are corrections (0-1)
    "dwell_cv",        # coefficient of variation of dwell (shape, not scale)
    "flight_cv",       # coefficient of variation of flight (rhythm consistency)
    "flight_p90",      # 90th-percentile flight (long-pause / hesitation marker)
    "rhythm_entropy",  # Shannon entropy of the flight distribution (rhythm structure)
]


def extract_features(dwell: list[float], flight: list[float],
                     backspace_rate: float = 0.0) -> dict:
    """Single source of truth: per-keystroke timing arrays (ms) -> 10-D vector."""
    d = np.asarray(dwell, dtype=float)
    f = np.asarray(flight, dtype=float)
    if d.size == 0:
        d = np.array([95.0])
    if f.size == 0:
        f = np.array([120.0])

    mean_d, std_d = float(d.mean()), float(d.std())
    mean_f, std_f = float(f.mean()), float(f.std())
    total_s = (d.sum() + np.clip(f, 0, None).sum()) / 1000.0
    speed = float(len(d) / total_s) if total_s > 0 else 4.0

    dwell_cv = std_d / mean_d if mean_d > 1e-6 else 0.0
    flight_cv = std_f / abs(mean_f) if abs(mean_f) > 1e-6 else 0.0
    flight_p90 = float(np.percentile(np.clip(f, 0, None), 90))

    # Shannon entropy of the flight distribution (8 bins over [0, p95]).
    fa = np.clip(f, 0, None)
    hi = np.percentile(fa, 95) or 1.0
    hist, _ = np.histogram(fa, bins=8, range=(0, hi))
    p = hist / hist.sum() if hist.sum() else np.ones(8) / 8
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    return {
        "mean_dwell": mean_d, "std_dwell": std_d,
        "mean_flight": mean_f, "std_flight": std_f,
        "typing_speed": speed, "backspace_rate": float(backspace_rate),
        "dwell_cv": dwell_cv, "flight_cv": flight_cv,
        "flight_p90": flight_p90, "rhythm_entropy": entropy,
    }


def vector(feat: dict) -> list[float]:
    return [float(feat.get(k, 0.0)) for k in FEATURES]


# ── Real training artifacts (CMU Keystroke Dynamics benchmark) ───────────────
_DATA = Path(__file__).resolve().parent / "model_data"
IMPOSTOR_POOL: np.ndarray | None = None
METRICS: dict = {}
POP_MEAN = np.array([95., 38., 125., 65., 4.4, .045, .4, .5, 200., 2.4])
POP_STD = np.array([28., 18., 55., 35., 1.7, .05, .2, .3, 110., .6])

try:
    IMPOSTOR_POOL = np.load(_DATA / "impostor_pool.npy").astype(float)
except Exception:
    IMPOSTOR_POOL = None
try:
    _pop = json.loads((_DATA / "population.json").read_text())
    if _pop.get("features") == FEATURES:
        POP_MEAN = np.array(_pop["mean"], dtype=float)
        POP_STD = np.maximum(np.array(_pop["std"], dtype=float), 1e-3)
except Exception:
    pass
try:
    METRICS = json.loads((_DATA / "metrics.json").read_text())
except Exception:
    METRICS = {}

DATA_BACKED = IMPOSTOR_POOL is not None and IMPOSTOR_POOL.shape[1] == len(FEATURES)
if IMPOSTOR_POOL is not None and IMPOSTOR_POOL.shape[1] != len(FEATURES):
    IMPOSTOR_POOL = None  # stale artifact from a different feature set


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _impostor_samples(mean: np.ndarray, std: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Negative class = REAL other-people keystroke vectors (CMU pool)."""
    rng = np.random.default_rng(seed)
    if IMPOSTOR_POOL is not None and len(IMPOSTOR_POOL) >= 10:
        idx = rng.choice(len(IMPOSTOR_POOL), size=n, replace=n > len(IMPOSTOR_POOL))
        s = IMPOSTOR_POOL[idx].copy()
        s += rng.normal(0.0, IMPOSTOR_POOL.std(axis=0) * 0.05, size=s.shape)
        return np.clip(s, 0.0, None)
    # synthetic fallback only if the dataset is missing
    std = np.maximum(std, 1e-3)
    out = []
    for _ in range(n):
        direction = rng.choice([-1.0, 1.0], size=len(mean))
        s = mean + direction * rng.uniform(1.8, 3.6, size=len(mean)) * std
        out.append(s)
    return np.clip(np.array(out), 0.0, None)


class UserModel:
    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.iso: IsolationForest | None = None
        self.xgb = None
        self.mean = POP_MEAN.copy()
        self.std = POP_STD.copy()
        self.pointer_mean: np.ndarray | None = None
        self.pointer_std: np.ndarray | None = None
        self.enrolled = False
        self.n_samples = 0
        self.lstm_base: tuple[float, float] | None = None   # (mean, std) of user's own recon error

    # ── Enrolment ────────────────────────────────────────────────────────
    def enroll(self, raw_windows: list[dict], pointer: list[list[float]] | None = None) -> dict:
        """raw_windows: list of {dwell:[...], flight:[...], backspace_rate}."""
        if len(raw_windows) < 3:
            self.enrolled = False
            self.n_samples = len(raw_windows)
            return {"enrolled": False, "samples": len(raw_windows), "reason": "need >= 3 windows"}

        feats = [extract_features(w.get("dwell", []), w.get("flight", []),
                                  w.get("backspace_rate", 0.0)) for w in raw_windows]
        X = np.atleast_2d(np.array([vector(f) for f in feats], dtype=float))

        # Personal LSTM rhythm baseline: how well the user's OWN sequences
        # reconstruct. Drift is then measured relative to this (robust to the
        # CMU-password vs free-text distribution shift).
        if _LSTM is not None:
            errs = [e for w in raw_windows
                    if (e := _LSTM.recon_error(w.get("dwell", []), w.get("flight", []))) is not None]
            if len(errs) >= 2:
                self.lstm_base = (float(np.mean(errs)), float(np.std(errs)) + 1e-6)

        self.mean = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), 1e-3)
        rng = np.random.default_rng(7)
        aug = [X] + [X + rng.normal(0.0, self.std * 0.18, size=X.shape) for _ in range(10)]
        Xa = np.clip(np.vstack(aug), 0.0, None)

        self.scaler = StandardScaler().fit(Xa)
        Xs = self.scaler.transform(Xa)
        self.iso = IsolationForest(n_estimators=140, contamination=0.06, random_state=0).fit(Xs)

        att = _impostor_samples(self.mean, self.std, n=len(Xa))
        atts = self.scaler.transform(att)
        Xtr = np.vstack([Xs, atts])
        ytr = np.r_[np.zeros(len(Xs)), np.ones(len(atts))]
        if _HAS_XGB:
            try:
                self.xgb = xgb.XGBClassifier(
                    n_estimators=90, max_depth=3, learning_rate=0.2,
                    subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", n_jobs=1)
                self.xgb.fit(Xtr, ytr)
            except Exception:
                self.xgb = None

        if pointer:
            P = np.atleast_2d(np.array(pointer, dtype=float))
            self.pointer_mean = P.mean(axis=0)
            self.pointer_std = np.maximum(P.std(axis=0), 1e-3)

        self.enrolled = True
        self.n_samples = int(X.shape[0])
        return {"enrolled": True, "samples": self.n_samples,
                "baseline_ms": {"dwell": round(float(self.mean[0]), 1),
                                "flight": round(float(self.mean[2]), 1),
                                "speed_kps": round(float(self.mean[4]), 2)}}

    # ── Scoring ──────────────────────────────────────────────────────────
    def score(self, dwell: list[float], flight: list[float], backspace_rate: float = 0.0) -> dict:
        feat = extract_features(dwell, flight, backspace_rate)
        x = np.array(vector(feat), dtype=float).reshape(1, -1)

        z = (x[0] - self.mean) / self.std
        # Statistical distance uses the robust CORE features (dwell/flight/speed/
        # backspace). The derived features (CV, percentile, entropy) are powerful
        # for the tree models but noisy as raw z-scores, so they're excluded here.
        stat = float(np.tanh(np.mean(np.abs(z[:6])) / 2.6))

        iso_anom = stat
        ato = stat
        shap = np.abs(z)
        if self.enrolled and self.scaler is not None:
            xs = self.scaler.transform(x)
            raw = float(self.iso.decision_function(xs)[0])
            iso_anom = float(_sigmoid(-7.0 * raw))
            if self.xgb is not None:
                try:
                    ato = float(self.xgb.predict_proba(xs)[0, 1])
                    shap = np.abs(self.xgb.get_booster().predict(
                        xgb.DMatrix(xs), pred_contribs=True)[0][:-1])
                except Exception:
                    ato = stat
            base = 0.25 * iso_anom + 0.50 * ato + 0.25 * stat
        else:
            base = min(1.0, stat * 1.25)

        # Personal LSTM temporal-drift: reconstruction error relative to the
        # user's OWN enrolled rhythm (std-units above their baseline). Robust to
        # the CMU-password vs free-text distribution shift.
        lstm_anom = None
        if _LSTM is not None and self.enrolled and self.lstm_base is not None:
            try:
                err = _LSTM.recon_error(dwell, flight)
                if err is not None:
                    z = (err - self.lstm_base[0]) / self.lstm_base[1]
                    lstm_anom = float(np.clip(z / 4.0, 0.0, 1.0))
            except Exception:
                lstm_anom = None

        anomaly = float(np.clip(0.80 * base + 0.20 * lstm_anom if lstm_anom is not None else base, 0.0, 1.0))
        key_score = float(np.clip(100.0 * (1.0 - anomaly), 0.0, 100.0))

        total = float(shap.sum()) or 1.0
        contrib = {FEATURES[i]: round(float(shap[i] / total * 100.0), 1) for i in range(len(FEATURES))}

        return {
            "key_score": round(key_score, 1),
            "anomaly": round(anomaly, 3),
            "iso_score": round(iso_anom, 3),
            "ato_prob": round(ato, 3),
            "lstm_anom": None if lstm_anom is None else round(lstm_anom, 3),
            "stat_distance": round(stat, 3),
            "contrib": contrib,
            "features": {k: round(float(feat[k]), 2) for k in FEATURES},
            "enrolled": self.enrolled,
        }

    def score_pointer(self, pointer: list[float] | None) -> float | None:
        if not pointer:
            return None
        p = np.array(pointer, dtype=float)
        if self.pointer_mean is None:
            return float(np.clip(90.0 - min(p[1] if len(p) > 1 else 0, 30.0), 55.0, 96.0))
        zp = (p - self.pointer_mean[:len(p)]) / self.pointer_std[:len(p)]
        d = float(np.tanh(np.mean(np.abs(zp)) / 2.6))
        return float(np.clip(100.0 * (1.0 - d), 0.0, 100.0))
