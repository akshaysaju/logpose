import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from forex_ml.config import Config
from forex_ml.data import generate_synthetic


def test_synthetic_shape_and_ohlc_validity():
    cfg = Config(n_synthetic=500, seed=1)
    df = generate_synthetic(cfg)
    assert len(df) == 500
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # High is the bar max, low is the bar min.
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12).all()
    assert (df["high"] >= df["low"]).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert df.index.is_monotonic_increasing


def test_synthetic_is_reproducible():
    a = generate_synthetic(Config(n_synthetic=200, seed=7))
    b = generate_synthetic(Config(n_synthetic=200, seed=7))
    assert np.allclose(a.to_numpy(), b.to_numpy())
