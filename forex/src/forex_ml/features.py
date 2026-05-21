"""Technical-indicator feature engineering.

Every feature is *causal*: the value at bar ``t`` uses only data from bars
``<= t``. pandas ``rolling``/``ewm`` are backward-looking by default, so no
lookahead is introduced here. The forward-looking target lives in ``labeling``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    feat = pd.DataFrame(index=df.index)

    # Returns
    feat["ret_1"] = close.pct_change()
    feat["log_ret_1"] = np.log(close).diff()
    for k in (3, 5, 10, 20):
        feat[f"mom_{k}"] = close.pct_change(k)

    # Moving-average relationships (scale-free)
    for w in (5, 10, 20, 50):
        sma = close.rolling(w).mean()
        feat[f"close_sma_{w}"] = close / sma - 1.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    feat["ema_12_26"] = ema12 / ema26 - 1.0

    # MACD
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd"] = macd / close
    feat["macd_signal"] = signal / close
    feat["macd_hist"] = (macd - signal) / close

    # RSI
    feat["rsi_14"] = _rsi(close, 14) / 100.0

    # Bollinger bands (20, 2)
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    feat["bb_pctb"] = (close - (ma20 - 2 * sd20)) / (4 * sd20)
    feat["bb_width"] = (4 * sd20) / ma20

    # Volatility
    feat["vol_10"] = feat["ret_1"].rolling(10).std()
    feat["vol_20"] = feat["ret_1"].rolling(20).std()
    feat["atr_14"] = _atr(df, 14) / close

    # Intrabar range / shape
    feat["hl_range"] = (df["high"] - df["low"]) / close
    feat["co_range"] = (close - df["open"]) / df["open"].replace(0.0, np.nan)

    # Volume (z-scored over a trailing window); 0 when volume is absent
    vol = df["volume"]
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0.0, np.nan)
    feat["vol_z"] = ((vol - vol_mean) / vol_std).fillna(0.0)

    return feat.replace([np.inf, -np.inf], np.nan)


def feature_names(df: pd.DataFrame) -> list[str]:
    return list(build_features(df).columns)
