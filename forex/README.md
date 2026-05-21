# forex-ml — Directional FX Prediction & Backtesting

A from-scratch machine-learning / deep-learning pipeline for **forex direction
prediction**. It engineers technical features from OHLC bars, trains classical
ML models **and** an LSTM to predict whether the next bar closes up or down, then
**backtests** the resulting trading signal with realistic transaction costs and
compares it to buy-and-hold.

## What's inside

```
forex/
├── run.py                       # CLI entry point
├── requirements.txt
├── src/forex_ml/
│   ├── config.py                # one dataclass with every tunable knob
│   ├── data.py                  # yfinance loader + offline synthetic generator + CSV cache
│   ├── features.py              # 23 causal technical indicators (RSI, MACD, Bollinger, ATR, …)
│   ├── labeling.py              # forward-return -> up/down label
│   ├── dataset.py               # chronological split, train-only scaling, LSTM windowing
│   ├── model_ml.py              # RandomForest + XGBoost (class-balanced)
│   ├── model_dl.py              # PyTorch LSTM classifier w/ early stopping
│   ├── backtest.py              # vectorized signal backtest (Sharpe, drawdown, win rate…)
│   ├── metrics.py               # accuracy / precision / recall / F1 / ROC-AUC
│   └── pipeline.py              # glues it all together + text report
└── tests/                       # pytest suite (data, features, leakage, backtest, pipeline)
```

## Quick start

```bash
cd forex
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py                       # synthetic data, all 3 models, prints a report + plot
```

Outputs: a comparison table in the terminal, `reports/results.json`, and
`reports/equity_curves.png` (out-of-sample equity curves vs buy-and-hold).

### Useful flags

```bash
python run.py --models random_forest xgboost      # skip the LSTM (much faster)
python run.py --strategy long_only --cost-bps 2   # long-only, 2 bps round-turn cost
python run.py --horizon 5 --window 48 --epochs 60 # predict 5 bars ahead, longer LSTM
python run.py --source yfinance --symbol EURUSD=X --start 2015-01-01   # real data (see below)
```

## Data sources

| Source        | Flag                     | Notes |
|---------------|--------------------------|-------|
| **synthetic** | default                  | Offline OHLC generator with GARCH-style volatility clustering. Zero network, fully reproducible. |
| **yfinance**  | `--source yfinance`      | Real historical FX bars from Yahoo Finance. |

> **Why synthetic is the default.** This repo was built in a sandbox whose
> network policy blocks Yahoo Finance (and every other market-data host). The
> `yfinance` path is fully implemented and works wherever Yahoo is reachable —
> just run it on your own machine. If a live download fails, the loader falls
> back to synthetic data so the pipeline never hard-crashes
> (`allow_synthetic_fallback`).

The synthetic series is a near-random walk **by design**, so models score
~0.50 accuracy / ~0.50 ROC-AUC on it. That is the *expected, healthy* result:
if a model scored 0.65 on a random walk, it would mean the pipeline was leaking
future information. The value here is the framework — point it at real data to
look for genuine (usually weak) structure.

## Design notes — no lookahead bias

Backtest realism lives or dies on leakage control. This pipeline enforces:

- **Causal features** — every indicator at bar `t` uses only bars `≤ t`
  (verified by `test_features.py::test_features_are_causal`).
- **Chronological split** — the test set is strictly later than train; no
  shuffling.
- **Train-only scaling** — the `StandardScaler` is fit on the train block, then
  applied to test.
- **Honest backtest timing** — a position decided at the close of bar `t` earns
  the return of bar `t+1`; turnover is charged a transaction cost.

## Models

- **Random Forest** & **XGBoost** — gradient-boosted / bagged trees on the 23
  engineered features, class-balanced for directional imbalance.
- **LSTM** (PyTorch) — 2-layer recurrent net over sliding feature windows, with
  a chronological validation tail for early stopping and class-weighted BCE.

## Tests

```bash
pytest -q        # 11 tests: data validity, feature causality, split integrity,
                 # backtest sanity (oracle vs costs), and end-to-end smoke tests
```
