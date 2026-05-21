#!/usr/bin/env python3
"""Command-line entry point for the forex ML/DL pipeline.

Examples:
    python run.py                              # synthetic data, all models
    python run.py --source yfinance --symbol EURUSD=X --start 2015-01-01
    python run.py --models random_forest xgboost --strategy long_only
    python run.py --no-plot --json results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from forex_ml.config import REPORTS_DIR, Config  # noqa: E402
from forex_ml.pipeline import format_report, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forex directional ML/DL pipeline")
    p.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--symbol", default="EURUSD=X")
    p.add_argument("--interval", default="1d")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--n-synthetic", type=int, default=3000)
    p.add_argument("--no-cache", action="store_true")

    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--label-threshold", type=float, default=0.0)
    p.add_argument("--test-size", type=float, default=0.2)

    p.add_argument("--window", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)

    p.add_argument("--strategy", choices=["long_only", "long_short"], default="long_short")
    p.add_argument("--cost-bps", type=float, default=1.0)

    p.add_argument(
        "--models", nargs="+",
        choices=["random_forest", "xgboost", "lstm"],
        default=["random_forest", "xgboost", "lstm"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default=None, help="write full results to this JSON path")
    p.add_argument("--no-plot", action="store_true")
    return p


def config_from_args(a: argparse.Namespace) -> Config:
    return Config(
        source=a.source, symbol=a.symbol, interval=a.interval, start=a.start, end=a.end,
        n_synthetic=a.n_synthetic, use_cache=not a.no_cache,
        horizon=a.horizon, label_threshold=a.label_threshold, test_size=a.test_size,
        window=a.window, epochs=a.epochs,
        strategy=a.strategy, cost_bps=a.cost_bps,
        models=tuple(a.models), seed=a.seed,
    )


def save_plot(out, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    if out.benchmark is not None:
        ax.plot(out.benchmark.equity.index, out.benchmark.equity.values,
                label="buy & hold", color="black", linewidth=1.4, linestyle="--")
    for name, bt in out.backtests.items():
        ax.plot(bt.equity.index, bt.equity.values, label=name, linewidth=1.4)
    ax.set_title("Out-of-sample equity curves")
    ax.set_ylabel("equity (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    out = run_pipeline(cfg)

    print()
    print(format_report(out))

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(out.results, indent=2))
        print(f"\nwrote results -> {args.json}")
    if not args.no_plot:
        target = REPORTS_DIR / "equity_curves.png"
        if save_plot(out, target):
            print(f"wrote plot    -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
