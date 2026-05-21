"""Central configuration for the forex ML/DL pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


@dataclass
class Config:
    # ---- data ----
    source: Literal["synthetic", "yfinance"] = "synthetic"
    symbol: str = "EURUSD=X"          # yfinance FX ticker
    interval: str = "1d"
    start: str = "2010-01-01"
    end: str | None = None
    n_synthetic: int = 3000           # bars when source == "synthetic"
    use_cache: bool = True
    # Fall back to synthetic data if a live download fails (e.g. host blocked).
    allow_synthetic_fallback: bool = True

    # ---- labeling ----
    horizon: int = 1                  # predict direction `horizon` bars ahead
    label_threshold: float = 0.0      # forward-return cutoff for an "up" label

    # ---- split / scaling ----
    test_size: float = 0.2            # final fraction held out chronologically
    val_size: float = 0.1             # fraction of the train block used for DL validation

    # ---- deep learning ----
    window: int = 32                  # sequence length for the LSTM
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 8                 # early-stopping patience (epochs)

    # ---- backtest ----
    strategy: Literal["long_only", "long_short"] = "long_short"
    cost_bps: float = 1.0             # round-turn cost per unit turnover, in basis points
    bars_per_year: int = 252          # annualization factor (252 for daily)
    long_threshold: float = 0.5       # P(up) above this -> long
    short_threshold: float = 0.5      # P(up) below this -> short (long_short only)

    # ---- misc ----
    seed: int = 42
    models: tuple[str, ...] = ("random_forest", "xgboost", "lstm")

    def to_dict(self) -> dict:
        return asdict(self)
