"""
Build real training artifacts from the CMU Keystroke Dynamics benchmark
(Killourhy & Maxion 2009 — 51 subjects x 400 reps of ".tie5Roanl").

Uses backend.engine.extract_features so the dataset, runtime and live capture
all compute identical 10-D vectors. Writes to backend/model_data/:
  impostor_pool.npy  — real other-people feature vectors (negative class)
  population.json     — real cold-start baseline stats
  metrics.json        — leakage-free evaluation (genuine vs unseen impostors)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from backend.engine import extract_features, vector, FEATURES  # noqa: E402

OUT = ROOT / "backend" / "model_data"
OUT.mkdir(parents=True, exist_ok=True)
CSV = HERE / "DSL-StrongPasswordData.csv"
rng = np.random.default_rng(20260622)


def load() -> dict[str, np.ndarray]:
    with open(CSV, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        h_idx = [i for i, c in enumerate(header) if c.startswith("H.")]
        ud_idx = [i for i, c in enumerate(header) if c.startswith("UD.")]
        s_idx = header.index("subject")
        rows: dict[str, list[list[float]]] = {}
        for r in reader:
            if not r:
                continue
            dwell = [float(r[i]) * 1000.0 for i in h_idx]      # s -> ms
            flight = [float(r[i]) * 1000.0 for i in ud_idx]
            backspace = float(np.clip(abs(rng.normal(0.04, 0.025)), 0, 0.2))
            rows.setdefault(r[s_idx], []).append(vector(extract_features(dwell, flight, backspace)))
    return {k: np.array(v) for k, v in rows.items()}


def roc_auc(y, p):
    order = np.argsort(-p); y = y[order]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return 0.5
    tpr = np.cumsum(y) / pos; fpr = np.cumsum(1 - y) / neg
    return float(np.trapezoid(tpr, fpr))


def eer(y, p):
    best, gap = 0.5, 1e9
    for t in np.linspace(0, 1, 200):
        far = ((p < t) & (y == 1)).sum() / max(1, (y == 1).sum())
        frr = ((p >= t) & (y == 0)).sum() / max(1, (y == 0).sum())
        if abs(far - frr) < gap:
            gap, best = abs(far - frr), (far + frr) / 2
    return float(best)


def evaluate(data: dict[str, np.ndarray]) -> dict:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    subjects = sorted(data)
    accs, aucs, eers = [], [], []
    for g in subjects:
        X = data[g]; tr_g, te_g = X[:200], X[200:]
        others = [s for s in subjects if s != g]
        tr_ids, te_ids = others[: len(others) * 4 // 5], others[len(others) * 4 // 5:]
        imp_tr = np.vstack([data[s][rng.choice(len(data[s]), 8, replace=False)] for s in tr_ids])
        imp_te = np.vstack([data[s][rng.choice(len(data[s]), 4, replace=False)] for s in te_ids])
        Xtr = np.vstack([tr_g, imp_tr]); ytr = np.r_[np.zeros(len(tr_g)), np.ones(len(imp_tr))]
        sc = StandardScaler().fit(Xtr)
        clf = xgb.XGBClassifier(n_estimators=90, max_depth=3, learning_rate=0.2,
                                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", n_jobs=1)
        clf.fit(sc.transform(Xtr), ytr)
        Xte = np.vstack([te_g, imp_te]); yte = np.r_[np.zeros(len(te_g)), np.ones(len(imp_te))]
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        accs.append(float(((p > 0.5).astype(int) == yte).mean()))
        aucs.append(roc_auc(yte, p)); eers.append(eer(yte, p))
    return {"subjects": len(subjects),
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "eer_mean": round(float(np.mean(eers)), 4),
            "features": len(FEATURES),
            "protocol": "per-subject genuine vs impostors from unseen identities; "
                        "genuine train=200 reps, impostor identities disjoint train/test",
            "dataset": "CMU DSL-StrongPasswordData (Killourhy & Maxion 2009), 51 subjects"}


def main() -> None:
    data = load()
    allX = np.vstack(list(data.values()))
    print(f"Loaded {len(data)} subjects, {len(allX)} samples, {len(FEATURES)} features.")
    for i, name in enumerate(FEATURES):
        print(f"  {name:16s} mean={allX[:,i].mean():8.2f} std={allX[:,i].std():8.2f}")

    idx = rng.choice(len(allX), size=min(2500, len(allX)), replace=False)
    np.save(OUT / "impostor_pool.npy", allX[idx].astype(np.float32))
    (OUT / "population.json").write_text(json.dumps({
        "mean": [round(float(x), 4) for x in allX.mean(axis=0)],
        "std": [round(float(x), 4) for x in allX.std(axis=0)],
        "features": FEATURES,
        "source": "CMU DSL-StrongPasswordData (Killourhy & Maxion 2009)"}, indent=2))

    print("\nRunning leakage-free evaluation (51 models)...")
    metrics = evaluate(data)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
