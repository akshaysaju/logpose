import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from forex_ml.config import Config
from forex_ml.data import generate_synthetic
from forex_ml.dataset import build_supervised, make_sequences, time_split
from forex_ml.labeling import make_labels


def test_labels_match_forward_return_sign():
    df = generate_synthetic(Config(n_synthetic=300, seed=4))
    label, fwd = make_labels(df, horizon=1, threshold=0.0)
    valid = fwd.dropna().index
    assert (label.loc[valid] == (fwd.loc[valid] > 0).astype(float)).all()
    # Last row has no future bar -> NaN label.
    assert np.isnan(label.iloc[-1])


def test_time_split_is_chronological_and_scaler_train_only():
    cfg = Config(n_synthetic=600, seed=5, test_size=0.25)
    df = generate_synthetic(cfg)
    data = build_supervised(df, cfg)
    sp = time_split(data, cfg)
    assert sp.train_idx.max() < sp.test_idx.min()
    assert len(sp.test_idx) == round(len(data.X) * cfg.test_size)
    # StandardScaler fit on train: train columns ~ zero mean / unit std.
    assert np.allclose(sp.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(sp.X_train.std(axis=0), 1.0, atol=1e-2)


def test_sequences_target_alignment():
    X = np.arange(20 * 3, dtype=float).reshape(20, 3)
    y = np.arange(20, dtype=float)
    seqs, targ, pos = make_sequences(X, y, window=5)
    assert seqs.shape == (16, 5, 3)
    # Each sequence ends at its target position.
    assert (targ == y[pos]).all()
    assert (seqs[0, -1] == X[4]).all()
    assert pos[0] == 4 and pos[-1] == 19
