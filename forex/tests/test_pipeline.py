import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forex_ml.config import Config
from forex_ml.pipeline import format_report, run_pipeline


def test_ml_pipeline_smoke():
    cfg = Config(
        source="synthetic", n_synthetic=800, use_cache=False,
        models=("random_forest", "xgboost"), test_size=0.25,
    )
    out = run_pipeline(cfg)
    for name in cfg.models:
        m = out.results["models"][name]
        assert 0.0 <= m["classification"]["accuracy"] <= 1.0
        assert "sharpe" in m["backtest"]
        assert name in out.backtests
    assert "buy & hold" not in out.results["models"]  # benchmark is separate
    assert isinstance(format_report(out), str)


def test_lstm_pipeline_smoke():
    from forex_ml.model_dl import torch_available
    if not torch_available():
        return  # torch optional
    cfg = Config(
        source="synthetic", n_synthetic=700, use_cache=False,
        models=("lstm",), window=16, epochs=3, test_size=0.25,
    )
    out = run_pipeline(cfg)
    m = out.results["models"]["lstm"]
    assert "classification" in m
    assert 0.0 <= m["classification"]["accuracy"] <= 1.0
