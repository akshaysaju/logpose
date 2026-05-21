"""Event-free vectorized backtest for a directional signal.

Timing convention (no lookahead): at the close of bar ``t`` we observe the model
probability, decide a position, and hold it through bar ``t+1``. The position
therefore earns ``fwd_ret[t] = close[t+1]/close[t] - 1``. Transaction cost is
charged on turnover whenever the position changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class BacktestResult:
    equity: pd.Series          # cumulative strategy equity (starts at 1.0)
    benchmark: pd.Series       # buy & hold equity
    positions: pd.Series       # -1 / 0 / +1 per bar
    returns: pd.Series         # per-bar net strategy return
    metrics: dict

    def summary(self) -> dict:
        return self.metrics


def _positions_from_proba(proba_up: np.ndarray, cfg: Config) -> np.ndarray:
    pos = np.zeros(len(proba_up))
    pos[proba_up >= cfg.long_threshold] = 1.0
    if cfg.strategy == "long_short":
        pos[proba_up < cfg.short_threshold] = -1.0
    return pos


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def run_backtest(
    proba_up: np.ndarray,
    fwd_ret: pd.Series,
    cfg: Config,
) -> BacktestResult:
    proba_up = np.asarray(proba_up, dtype=float)
    fwd = fwd_ret.to_numpy(dtype=float)
    idx = fwd_ret.index

    positions = _positions_from_proba(proba_up, cfg)
    # Turnover at bar t = |pos_t - pos_{t-1}|; first bar opens from flat.
    prev = np.concatenate([[0.0], positions[:-1]])
    turnover = np.abs(positions - prev)
    cost = turnover * (cfg.cost_bps / 1e4)

    gross = positions * fwd
    net = gross - cost

    equity = np.cumprod(1.0 + net)
    benchmark = np.cumprod(1.0 + fwd)

    ann = cfg.bars_per_year
    mean, std = net.mean(), net.std(ddof=1) if len(net) > 1 else 0.0
    sharpe = float(np.sqrt(ann) * mean / std) if std > 0 else float("nan")

    n_years = len(net) / ann
    cagr = float(equity[-1] ** (1 / n_years) - 1.0) if n_years > 0 and equity[-1] > 0 else float("nan")
    traded = net[positions != 0.0]
    metrics = {
        "total_return": float(equity[-1] - 1.0),
        "benchmark_return": float(benchmark[-1] - 1.0),
        "cagr": cagr,
        "ann_volatility": float(std * np.sqrt(ann)),
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "win_rate": float((traded > 0).mean()) if len(traded) else float("nan"),
        "n_trades": int((turnover > 0).sum()),
        "exposure": float((positions != 0.0).mean()),
        "n_bars": int(len(net)),
    }

    return BacktestResult(
        equity=pd.Series(equity, index=idx, name="equity"),
        benchmark=pd.Series(benchmark, index=idx, name="benchmark"),
        positions=pd.Series(positions, index=idx, name="position"),
        returns=pd.Series(net, index=idx, name="net_return"),
        metrics=metrics,
    )
