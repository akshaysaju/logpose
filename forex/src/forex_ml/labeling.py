"""Target construction for directional prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(
    df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0
) -> tuple[pd.Series, pd.Series]:
    """Return ``(label, forward_return)`` aligned to ``df.index``.

    ``forward_return[t]`` is the return from ``close[t]`` to ``close[t+horizon]``.
    ``label[t]`` is 1 when that forward return exceeds ``threshold`` else 0.
    The final ``horizon`` rows have NaN (no future bar) and should be dropped.
    """
    close = df["close"]
    fwd_ret = close.shift(-horizon) / close - 1.0
    label = (fwd_ret > threshold).astype(float)
    label[fwd_ret.isna()] = np.nan
    return label.rename("label"), fwd_ret.rename("fwd_ret")
