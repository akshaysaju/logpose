"""Assemble supervised data, split it chronologically, scale, and window.

Anti-lookahead guarantees enforced here:
  * the chronological split never shuffles, so test bars are strictly later
    than train bars;
  * the scaler is fit on the training block only;
  * LSTM sequences are assigned to train/test by the *target* bar's position, so
    no test target is ever predicted from features that post-date it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import build_features
from .labeling import make_labels


@dataclass
class Supervised:
    X: pd.DataFrame          # feature matrix, NaNs dropped
    y: pd.Series             # 0/1 direction labels
    fwd_ret: pd.Series       # realized forward return for each row
    close: pd.Series         # close price aligned to X.index
    feature_names: list[str]


def build_supervised(df: pd.DataFrame, cfg: Config) -> Supervised:
    feats = build_features(df)
    label, fwd_ret = make_labels(df, cfg.horizon, cfg.label_threshold)

    joined = feats.copy()
    joined["__label__"] = label
    joined["__fwd__"] = fwd_ret
    joined["__close__"] = df["close"]
    joined = joined.dropna()

    X = joined.drop(columns=["__label__", "__fwd__", "__close__"])
    return Supervised(
        X=X,
        y=joined["__label__"].astype(int),
        fwd_ret=joined["__fwd__"],
        close=joined["__close__"],
        feature_names=list(X.columns),
    )


@dataclass
class Split:
    train_idx: pd.Index
    test_idx: pd.Index
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def time_split(data: Supervised, cfg: Config) -> Split:
    n = len(data.X)
    n_test = max(1, int(round(n * cfg.test_size)))
    split = n - n_test

    X_all = data.X.to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X_all[:split])

    return Split(
        train_idx=data.X.index[:split],
        test_idx=data.X.index[split:],
        X_train=scaler.transform(X_all[:split]),
        X_test=scaler.transform(X_all[split:]),
        y_train=data.y.to_numpy()[:split],
        y_test=data.y.to_numpy()[split:],
        scaler=scaler,
    )


def make_sequences(
    X: np.ndarray, y: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows ``X[t-window+1 .. t]`` -> target ``y[t]``.

    Returns ``(sequences, targets, target_positions)`` where ``target_positions``
    indexes back into the original arrays so callers can map a sequence to its
    train/test side of the chronological split.
    """
    if len(X) <= window:
        return (
            np.empty((0, window, X.shape[1])),
            np.empty((0,)),
            np.empty((0,), dtype=int),
        )
    n_seq = len(X) - window + 1
    seqs = np.stack([X[i : i + window] for i in range(n_seq)])
    positions = np.arange(window - 1, len(X))
    return seqs, y[positions], positions
