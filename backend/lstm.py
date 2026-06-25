"""
ZeroLeakX — real LSTM temporal-drift detector.

A PyTorch LSTM *autoencoder* over the per-keystroke timing sequence
([dwell, preceding-flight] per key). Trained on genuine human typing sequences
from the CMU benchmark to reconstruct normal rhythm; a session that does not
reconstruct well (high error) is temporally anomalous. This is the diagram's
"LSTM" box, implemented for real (replacing the earlier EWMA stand-in, which is
still kept in the session layer as a complementary session-level drift signal).

Weights are built offline by data/train_lstm.py -> backend/model_data/lstm.pt
(+ lstm_norm.json). If torch or the weights are missing, load_detector() returns
None and the engine simply omits the LSTM term — graceful degradation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "model_data"

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:
    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_feat: int = 2, hidden: int = 16):
            super().__init__()
            self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
            self.dec = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_feat)

        def forward(self, x):                       # x: (B, T, F)
            _, (h, _) = self.enc(x)
            T = x.size(1)
            z = h[-1].unsqueeze(1).repeat(1, T, 1)  # latent repeated over time
            d, _ = self.dec(z)
            return self.out(d)


def build_sequence(dwell: list[float], flight: list[float]) -> np.ndarray:
    """Per-key steps [dwell_i, flight_{i-1}] (ms). Length = #keys."""
    d = np.asarray(dwell, dtype=float)
    if d.size == 0:
        return np.zeros((0, 2))
    fprev = np.zeros_like(d)
    f = np.asarray(flight, dtype=float)
    if f.size:
        fprev[1:1 + len(f)] = np.clip(f[: len(d) - 1], 0, None)
    return np.stack([d, fprev], axis=1)            # (T, 2)


class Detector:
    def __init__(self, model, mean, std, p50, p95):
        self.model = model
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.maximum(np.asarray(std, dtype=float), 1e-6)
        self.p50, self.p95 = float(p50), float(p95)

    def recon_error(self, dwell: list[float], flight: list[float]) -> float | None:
        """Raw reconstruction error for a sequence (None if too short)."""
        seq = build_sequence(dwell, flight)
        if seq.shape[0] < 2:
            return None
        seq = seq[:24]
        x = (seq - self.mean) / self.std
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rec = self.model(xt).squeeze(0).numpy()
        return float(np.mean((rec - x) ** 2))

    def anomaly(self, dwell: list[float], flight: list[float]) -> float:
        """Population-calibrated anomaly (used only before a personal baseline)."""
        err = self.recon_error(dwell, flight)
        if err is None:
            return 0.0
        return float(np.clip((err - self.p50) / (self.p95 - self.p50 + 1e-9), 0.0, 1.0))


def load_detector():
    if not _HAS_TORCH:
        return None
    try:
        norm = json.loads((_DATA / "lstm_norm.json").read_text())
        model = LSTMAutoencoder(n_feat=2, hidden=norm.get("hidden", 16))
        model.load_state_dict(torch.load(_DATA / "lstm.pt", map_location="cpu"))
        model.eval()
        return Detector(model, norm["mean"], norm["std"], norm["p50"], norm["p95"])
    except Exception:
        return None
