"""End-to-end pipeline: load -> features -> split -> train -> evaluate -> backtest."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config
from .data import load_data
from .dataset import build_supervised, make_sequences, time_split
from .metrics import classification_metrics
from .model_dl import LSTMModel, torch_available
from .model_ml import ML_TRAINERS, predict_proba_up

log = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    results: dict
    backtests: dict[str, BacktestResult] = field(default_factory=dict)
    benchmark: BacktestResult | None = None


def _evaluate(name: str, proba_up: np.ndarray, y_test: np.ndarray, fwd_test: pd.Series, cfg: Config):
    clf = classification_metrics(y_test, proba_up)
    bt = run_backtest(proba_up, fwd_test, cfg)
    log.info(
        "%-14s acc=%.3f auc=%.3f | ret=%+.1f%% sharpe=%.2f mdd=%.1f%%",
        name, clf["accuracy"], clf["roc_auc"],
        100 * bt.metrics["total_return"], bt.metrics["sharpe"],
        100 * bt.metrics["max_drawdown"],
    )
    return clf, bt


def run_pipeline(cfg: Config) -> PipelineOutput:
    np.random.seed(cfg.seed)

    df = load_data(cfg)
    data = build_supervised(df, cfg)
    split = time_split(data, cfg)

    fwd = data.fwd_ret
    fwd_test = fwd.loc[split.test_idx]

    results: dict = {
        "config": cfg.to_dict(),
        "data": {
            "source": cfg.source,
            "n_bars": int(len(df)),
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "n_features": len(data.feature_names),
            "feature_names": data.feature_names,
        },
        "split": {"n_train": int(len(split.train_idx)), "n_test": int(len(split.test_idx))},
        "models": {},
    }
    out = PipelineOutput(results=results)

    # Buy & hold benchmark on the test window (always long).
    out.benchmark = run_backtest(np.ones(len(fwd_test)), fwd_test, cfg)
    results["benchmark"] = out.benchmark.metrics

    requested = set(cfg.models)

    # --- classical ML ---
    for name, trainer in ML_TRAINERS.items():
        if name not in requested:
            continue
        model = trainer(split.X_train, split.y_train, cfg)
        proba = predict_proba_up(model, split.X_test)
        clf, bt = _evaluate(name, proba, split.y_test, fwd_test, cfg)
        entry = {"classification": clf, "backtest": bt.metrics}
        if hasattr(model, "feature_importances_"):
            imp = sorted(
                zip(data.feature_names, model.feature_importances_.tolist()),
                key=lambda kv: kv[1], reverse=True,
            )
            entry["top_features"] = imp[:8]
        results["models"][name] = entry
        out.backtests[name] = bt

    # --- LSTM ---
    if "lstm" in requested:
        if not torch_available():
            log.warning("torch unavailable; skipping LSTM")
            results["models"]["lstm"] = {"skipped": "torch not installed"}
        else:
            X_all = split.scaler.transform(data.X.to_numpy(dtype=np.float64))
            y_all = data.y.to_numpy()
            seqs, seq_y, positions = make_sequences(X_all, y_all, cfg.window)
            n_train = len(split.train_idx)
            tr = positions < n_train
            te = positions >= n_train
            if te.sum() == 0 or tr.sum() == 0:
                results["models"]["lstm"] = {"skipped": "not enough bars for the configured window"}
            else:
                lstm = LSTMModel(cfg).fit(seqs[tr], seq_y[tr])
                proba = lstm.predict_proba_up(seqs[te])
                # Sequence targets line up with the tail of the test index.
                fwd_seq_test = fwd.iloc[positions[te]]
                clf, bt = _evaluate("lstm", proba, y_all[positions[te]], fwd_seq_test, cfg)
                results["models"]["lstm"] = {"classification": clf, "backtest": bt.metrics}
                out.backtests["lstm"] = bt

    return out


def format_report(out: PipelineOutput) -> str:
    r = out.results
    lines = []
    lines.append("=" * 64)
    lines.append("  FOREX DIRECTIONAL MODEL — REPORT")
    lines.append("=" * 64)
    d = r["data"]
    lines.append(
        f"data: {d['source']}  bars={d['n_bars']}  {d['start']}..{d['end']}  "
        f"features={d['n_features']}"
    )
    lines.append(f"split: train={r['split']['n_train']}  test={r['split']['n_test']}")
    b = r["benchmark"]
    lines.append(
        f"buy&hold (test): ret={100*b['total_return']:+.1f}%  "
        f"sharpe={b['sharpe']:.2f}  mdd={100*b['max_drawdown']:.1f}%"
    )
    lines.append("-" * 64)
    header = f"{'model':<14}{'acc':>7}{'auc':>7}{'ret%':>9}{'sharpe':>8}{'mdd%':>8}{'trades':>8}"
    lines.append(header)
    lines.append("-" * 64)
    for name, m in r["models"].items():
        if "classification" not in m:
            lines.append(f"{name:<14}  (skipped: {m.get('skipped','?')})")
            continue
        c, bt = m["classification"], m["backtest"]
        lines.append(
            f"{name:<14}{c['accuracy']:>7.3f}{c['roc_auc']:>7.3f}"
            f"{100*bt['total_return']:>9.1f}{bt['sharpe']:>8.2f}"
            f"{100*bt['max_drawdown']:>8.1f}{bt['n_trades']:>8d}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)
