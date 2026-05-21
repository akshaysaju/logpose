import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from forex_ml.config import Config
from forex_ml.backtest import run_backtest


def test_perfect_foresight_beats_random_and_costs_reduce_return():
    rng = np.random.default_rng(0)
    fwd = pd.Series(rng.normal(0, 0.01, 500))
    # A signal that perfectly knows the next move (proba 1 up, 0 down).
    oracle = (fwd > 0).astype(float).to_numpy()

    free = run_backtest(oracle, fwd, Config(strategy="long_short", cost_bps=0.0))
    costed = run_backtest(oracle, fwd, Config(strategy="long_short", cost_bps=5.0))

    assert free.metrics["total_return"] > 0
    assert free.metrics["win_rate"] == 1.0
    # Transaction costs can only erode a given signal's return.
    assert costed.metrics["total_return"] <= free.metrics["total_return"]
    assert len(free.equity) == len(fwd)


def test_flat_signal_has_no_trades_and_no_pnl():
    fwd = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
    # proba == 0.5 with long_only thresholds -> never long, never short.
    res = run_backtest(np.full(100, 0.49), fwd, Config(strategy="long_only", cost_bps=2.0))
    assert res.metrics["n_trades"] == 0
    assert abs(res.metrics["total_return"]) < 1e-12
