"""
Train the ZeroLeakX LSTM temporal-drift autoencoder on REAL keystroke sequences
from the CMU benchmark, and validate that it separates genuine from impostor
sequences. Writes backend/model_data/lstm.pt + lstm_norm.json.

Per typing sample -> sequence of per-key [dwell, preceding-flight] steps. The
autoencoder learns to reconstruct genuine human rhythm; reconstruction error is
the temporal anomaly score.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from backend.lstm import LSTMAutoencoder, build_sequence  # noqa: E402

OUT = ROOT / "backend" / "model_data"
CSV = HERE / "DSL-StrongPasswordData.csv"
HIDDEN = 16
torch.manual_seed(0)
np.random.seed(0)


def load_sequences():
    with open(CSV, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        h_idx = [i for i, c in enumerate(header) if c.startswith("H.")]
        ud_idx = [i for i, c in enumerate(header) if c.startswith("UD.")]
        s_idx = header.index("subject")
        per_subj: dict[str, list[np.ndarray]] = {}
        for r in reader:
            if not r:
                continue
            dwell = [float(r[i]) * 1000.0 for i in h_idx]
            flight = [float(r[i]) * 1000.0 for i in ud_idx]
            per_subj.setdefault(r[s_idx], []).append(build_sequence(dwell, flight))
    return per_subj


def main():
    per_subj = load_sequences()
    subjects = sorted(per_subj)
    seqs = [s for subj in subjects for s in per_subj[subj]]
    X = np.stack(seqs)                      # (N, 11, 2)
    N, T, F = X.shape
    print(f"{N} sequences, length {T}, {F} features/step.")

    mean = X.reshape(-1, F).mean(axis=0)
    std = X.reshape(-1, F).std(axis=0) + 1e-6
    Xn = (X - mean) / std

    # Train the autoencoder on genuine sequences only (unsupervised).
    train = torch.tensor(Xn, dtype=torch.float32)
    model = LSTMAutoencoder(n_feat=F, hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()
    model.train()
    bs = 256
    for epoch in range(12):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = train[idx]
            opt.zero_grad()
            rec = model(xb)
            loss = loss_fn(rec, xb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"  epoch {epoch+1:2d}  loss={tot/N:.4f}")

    # Per-sequence reconstruction error on the training (genuine) set.
    model.eval()
    with torch.no_grad():
        rec = model(train).numpy()
    err = ((rec - Xn) ** 2).mean(axis=(1, 2))
    p50, p95 = float(np.percentile(err, 50)), float(np.percentile(err, 95))

    torch.save(model.state_dict(), OUT / "lstm.pt")
    (OUT / "lstm_norm.json").write_text(json.dumps({
        "mean": [float(x) for x in mean], "std": [float(x) for x in std],
        "p50": p50, "p95": p95, "hidden": HIDDEN,
        "note": "LSTM autoencoder; anomaly = recon error scaled between p50..p95 of genuine errors"}, indent=2))

    # Sanity check: genuine vs cross-subject impostor error separation.
    from backend.lstm import Detector
    det = Detector(model, mean, std, p50, p95)
    g = subjects[0]
    gen = [det.anomaly(list(s[:, 0]), list(s[1:, 1])) for s in per_subj[g][:100]]
    imp = [det.anomaly(list(s[:, 0]), list(s[1:, 1])) for sj in subjects[1:11] for s in per_subj[sj][:10]]
    print(f"\nGenuine mean anomaly  = {np.mean(gen):.3f}")
    print(f"Impostor mean anomaly = {np.mean(imp):.3f}")
    print(f"Saved lstm.pt + lstm_norm.json")


if __name__ == "__main__":
    main()
