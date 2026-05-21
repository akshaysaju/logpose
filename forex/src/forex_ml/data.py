"""FX data loading.

Two sources:
  * ``yfinance`` — real historical FX bars (works wherever Yahoo Finance is
    reachable; some sandboxed networks block it).
  * ``synthetic`` — an offline OHLC generator with volatility clustering, so the
    full pipeline runs and is testable without any network access.

Loaded frames are cached as CSV under ``data/`` keyed by their parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, DATA_DIR

log = logging.getLogger(__name__)

OHLC_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(cfg: Config) -> Path:
    if cfg.source == "synthetic":
        key = f"synthetic_{cfg.n_synthetic}_{cfg.seed}"
    else:
        key = f"{cfg.symbol}_{cfg.interval}_{cfg.start}_{cfg.end}"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in key)
    return DATA_DIR / f"{safe}.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame to lowercase OHLC columns + DatetimeIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) columns when given a single symbol too.
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"data missing required columns: {missing} (got {list(df.columns)})")
    df = df[OHLC_COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.dropna().sort_index()


def load_yfinance(cfg: Config) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        cfg.symbol,
        interval=cfg.interval,
        start=cfg.start,
        end=cfg.end,
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no rows for {cfg.symbol!r}. "
            "The host may be blocked, or the symbol/interval is invalid."
        )
    return _normalize(raw)


def generate_synthetic(cfg: Config) -> pd.DataFrame:
    """Simulate realistic daily FX OHLC bars.

    Uses a GARCH(1,1)-style stochastic-volatility process so the series shows
    volatility clustering rather than constant-variance Gaussian noise.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_synthetic
    intrabar_steps = 24
    drift = 1e-5

    omega, alpha, beta = 2e-7, 0.07, 0.90
    var = omega / (1.0 - alpha - beta)

    price = 1.10
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for _ in range(n):
        bar_open = price
        path = np.empty(intrabar_steps)
        for s in range(intrabar_steps):
            eps = rng.standard_normal()
            shock = np.sqrt(var) * eps
            price *= np.exp(drift + shock)
            path[s] = price
            var = omega + alpha * shock**2 + beta * var
        opens.append(bar_open)
        closes.append(price)
        highs.append(max(bar_open, path.max()))
        lows.append(min(bar_open, path.min()))
        # Volume loosely scales with the bar's realized range.
        rng_pct = (highs[-1] - lows[-1]) / bar_open
        volumes.append(float(rng.integers(5_000, 20_000) * (1.0 + 50.0 * rng_pct)))

    idx = pd.bdate_range(start=cfg.start, periods=n)
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )
    df.index.name = "date"
    return df


def load_data(cfg: Config) -> pd.DataFrame:
    cache = _cache_path(cfg)
    if cfg.use_cache and cache.exists():
        log.info("loading cached data from %s", cache)
        return _normalize(pd.read_csv(cache, index_col=0, parse_dates=True))

    if cfg.source == "yfinance":
        try:
            df = load_yfinance(cfg)
        except Exception as exc:  # network blocked, bad symbol, etc.
            if not cfg.allow_synthetic_fallback:
                raise
            log.warning("yfinance load failed (%s); falling back to synthetic data", exc)
            df = generate_synthetic(cfg)
    elif cfg.source == "synthetic":
        df = generate_synthetic(cfg)
    else:
        raise ValueError(f"unknown data source: {cfg.source!r}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df
