"""LSTM directional classifier (PyTorch, CPU-friendly).

Trains on sliding windows of scaled features to predict P(next move is up).
Uses a chronological validation tail for early stopping and class-weighted
BCE to cope with directional imbalance.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _build_module(n_features: int, cfg: Config):
    import torch
    from torch import nn

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)  # logit per sequence

    return LSTMClassifier()


class LSTMModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.module = None
        self._device = "cpu"

    def fit(self, seqs: np.ndarray, targets: np.ndarray) -> "LSTMModel":
        import torch
        from torch import nn

        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        n = len(seqs)
        n_val = max(1, int(round(n * cfg.val_size)))
        tr_x, tr_y = seqs[: n - n_val], targets[: n - n_val]
        va_x, va_y = seqs[n - n_val :], targets[n - n_val :]

        device = self._device
        self.module = _build_module(seqs.shape[2], cfg).to(device)

        pos = float(tr_y.mean())
        pos = min(max(pos, 1e-6), 1 - 1e-6)
        pos_weight = torch.tensor([(1 - pos) / pos], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.module.parameters(), lr=cfg.lr, weight_decay=1e-5)

        tr_x_t = torch.tensor(tr_x, dtype=torch.float32, device=device)
        tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=device)
        va_x_t = torch.tensor(va_x, dtype=torch.float32, device=device)
        va_y_t = torch.tensor(va_y, dtype=torch.float32, device=device)

        best_val = float("inf")
        best_state = None
        wait = 0
        for _ in range(cfg.epochs):
            self.module.train()
            perm = torch.randperm(len(tr_x_t))
            for start in range(0, len(perm), cfg.batch_size):
                batch = perm[start : start + cfg.batch_size]
                opt.zero_grad()
                logits = self.module(tr_x_t[batch])
                loss = loss_fn(logits, tr_y_t[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
                opt.step()

            self.module.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.module(va_x_t), va_y_t).item()
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        return self

    def predict_proba_up(self, seqs: np.ndarray) -> np.ndarray:
        import torch

        if self.module is None:
            raise RuntimeError("model is not trained")
        if len(seqs) == 0:
            return np.empty((0,))
        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.tensor(seqs, dtype=torch.float32, device=self._device))
            return torch.sigmoid(logits).cpu().numpy()
