import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from forex_ml.config import Config
from forex_ml.data import generate_synthetic
from forex_ml.features import build_features


def test_features_have_no_nan_or_inf_after_warmup():
    df = generate_synthetic(Config(n_synthetic=400, seed=2))
    feats = build_features(df).dropna()
    assert len(feats) > 300
    assert np.isfinite(feats.to_numpy()).all()


def test_features_are_causal():
    """A feature at bar t must not change when future bars are appended."""
    df = generate_synthetic(Config(n_synthetic=400, seed=3))
    cutoff = 300
    full = build_features(df)
    truncated = build_features(df.iloc[:cutoff])
    common = full.index[:cutoff][-50:]  # compare a well-warmed-up tail
    a = full.loc[common]
    b = truncated.loc[common]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
