import numpy as np
import pandas as pd
import pytest

from cryptosent.config import Settings
from cryptosent.models.evaluate import backtest, regression_metrics
from cryptosent.models.registry import CorrelationFilter, make_model
from cryptosent.models.train import walk_forward_splits
from cryptosent.pipeline import predict_latest, run_demo
from cryptosent.storage import LocalStore


def test_regression_metrics_direction():
    m = regression_metrics(np.array([1.0, -1.0, 2.0, 0.0]), np.array([0.5, -0.2, -1.0, 0.3]))
    assert m["direction_accuracy"] == pytest.approx(2 / 3)


def test_backtest_costs_and_threshold():
    y = pd.Series([0.5, -0.5, 0.1, 0.05])
    pred = pd.Series([1.0, -1.0, 0.1, -0.05])
    bt = backtest(y, pred, pred_min=10, step_min=10, fee_pct=0.1)
    assert bt["trades"] == 2  # only |pred| > 0.2 round-trip fees
    assert bt["total_return_pct"] == pytest.approx(0.5 + 0.5 - 0.4)
    # overlapping samples are thinned to non-overlapping trades
    assert backtest(y, pred, pred_min=20, step_min=10)["periods"] == 2


def test_walk_forward_has_embargo():
    idx = pd.date_range("2025-01-01", periods=1000, freq="10min", tz="UTC")
    gap = pd.Timedelta(minutes=50)
    for tr, te in walk_forward_splits(idx, 4, 0.4, gap):
        assert idx[tr].max() < idx[te].min() - gap + pd.Timedelta(minutes=10)
        assert idx[tr].max() < idx[te].min()


def test_correlation_filter():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.c_[a, a * 2 + 0.001, rng.normal(size=200), np.ones(200)]
    assert list(CorrelationFilter(0.95).fit(X).keep_) == [0, 2]


@pytest.mark.parametrize("name", ["ridge", "random_forest", "gradient_boosting", "lightgbm"])
def test_models_fit(name):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("abcde"))
    y = X["a"] * 0.5 + rng.normal(scale=0.1, size=300)
    pred = make_model(name).fit(X[:250], y[:250]).predict(X[250:])
    assert regression_metrics(y[250:].to_numpy(), pred)["r2"] > 0.3


def test_demo_end_to_end_recovers_planted_news_signal(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts", mongo_uri=None)
    store = LocalStore(settings.data_dir)
    report = run_demo(settings, store, days=21, models=["ridge", "random_forest"])
    assert report["sanity"]["timestamp_audit"]["article_after_decision_time"] == 0
    # news carries the planted signal, price history alone does not
    news = report["cv_ablation"]["news_only"]
    tech = report["cv_ablation"]["technical_only"]
    assert news["information_coefficient_mean"] > tech["information_coefficient_mean"] + 0.05
    assert report["sanity"]["permutation"]["r2_shuffled_mean"] < 0.02
    assert (tmp_path / "artifacts" / "report.md").exists()
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert store.latest_model_run()["best_model"] in ("ridge", "random_forest")


def test_predict_latest_writes_predictions(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts", mongo_uri=None)
    store = LocalStore(settings.data_dir)
    run_demo(settings, store, days=10, models=["ridge"])
    last = store.get_prices().index.max().to_pydatetime()
    rows = predict_latest(settings, store, lookback_hours=3, now=last)
    assert rows and all(r["signal"] in ("long", "short", "flat") for r in rows)
    assert rows[-1]["actual_pct"] is None  # future not known yet
    assert len(store.get_predictions()) >= len(rows)
