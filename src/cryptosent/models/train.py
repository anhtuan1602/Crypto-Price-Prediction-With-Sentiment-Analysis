"""Experiment runner: purged walk-forward CV, chronological holdout, ablation,
baselines, sanity checks, backtest, MLflow logging and the final live model.

Protocol (fixes report sections 5.3 to 5.4):
  1. Hold out the last ``test_frac`` of time as the final test set.
  2. On the rest, run expanding-window walk-forward CV with an embargo of
     obs + lag + pred minutes between train and test so overlapping windows
     cannot leak. Pick the best model by mean CV MAE.
  3. Ablation of feature groups for that model (technical / news / both).
  4. Refit on the development period, evaluate once on the test set:
     metrics, baselines, permutation test, backtest.
  5. Refit on everything and save for live prediction.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..features.dataset import TARGET, Dataset, audit_dataset
from .baselines import baseline_predictions
from .evaluate import backtest, regression_metrics
from .registry import MODEL_NAMES, feature_importance, make_model

log = logging.getLogger(__name__)

FEATURE_SETS = {
    "technical_only": ["technical"],
    "news_only": ["sentiment", "nlp"],
    "technical_plus_news": ["technical", "sentiment", "nlp"],
    "all_with_time": ["technical", "sentiment", "nlp", "time"],
}
FULL_SET = "technical_plus_news"


@dataclass
class ExperimentConfig:
    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    n_splits: int = 5
    test_frac: float = 0.15
    min_train_frac: float = 0.4
    noise: float = 0.05
    random_state: int = 42
    fee_pct: float = 0.1
    permutation_runs: int = 3


def embargo(ds: Dataset) -> pd.Timedelta:
    w = ds.windows
    return pd.Timedelta(minutes=w.obs + w.lag + w.pred)


def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int, min_train_frac: float, gap: pd.Timedelta):
    """Expanding-window folds over ``index`` with a purge gap before each test block."""
    n = len(index)
    start = int(n * min_train_frac)
    bounds = np.linspace(start, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        a, b = bounds[k], bounds[k + 1]
        if b - a < 10:
            continue
        test_start = index[a]
        train_mask = index < (test_start - gap)
        if train_mask.sum() < 50:
            continue
        yield np.where(train_mask)[0], np.arange(a, b)


def _fit_predict(name: str, X_tr, y_tr, X_te, cfg: ExperimentConfig):
    model = make_model(name, cfg.random_state, cfg.noise)
    model.fit(X_tr, y_tr)
    return model, model.predict(X_te)


def cross_validate(df: pd.DataFrame, features: list[str], name: str, cfg: ExperimentConfig, gap: pd.Timedelta) -> dict[str, Any]:
    fold_metrics = []
    for tr, te in walk_forward_splits(df.index, cfg.n_splits, cfg.min_train_frac, gap):
        _, pred = _fit_predict(name, df.iloc[tr][features], df.iloc[tr][TARGET], df.iloc[te][features], cfg)
        fold_metrics.append(regression_metrics(df.iloc[te][TARGET].to_numpy(), pred))
    if not fold_metrics:
        return {"folds": 0}
    keys = [k for k in fold_metrics[0] if k != "n"]
    summary: dict[str, Any] = {"folds": len(fold_metrics)}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        summary[f"{k}_mean"] = float(np.nanmean(vals))
        summary[f"{k}_std"] = float(np.nanstd(vals))
    return summary


def autocorrelation(y: pd.Series, lags=(1, 2, 3)) -> dict[str, float]:
    return {f"lag_{k}": float(y.autocorr(k)) for k in lags}


def permutation_test(name, X_tr, y_tr, X_te, y_te, cfg: ExperimentConfig) -> dict[str, float]:
    rng = np.random.default_rng(cfg.random_state)
    scores = []
    for _ in range(cfg.permutation_runs):
        shuffled = rng.permutation(np.asarray(y_tr))
        _, pred = _fit_predict(name, X_tr, shuffled, X_te, cfg)
        scores.append(regression_metrics(np.asarray(y_te), pred)["r2"])
    return {"r2_shuffled_mean": float(np.mean(scores)), "runs": len(scores)}


def run_experiment(ds: Dataset, cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ExperimentConfig()
    df = ds.frame.sort_index()
    if len(df) < 200:
        raise ValueError(f"Need at least 200 labelled samples, got {len(df)}")
    gap = embargo(ds)
    audit = audit_dataset(ds)

    n_test = max(30, int(len(df) * cfg.test_frac))
    test_start = df.index[-n_test]
    dev = df[df.index < test_start - gap]
    test = df.iloc[-n_test:]
    log.info("dev=%d test=%d (from %s), embargo=%s", len(dev), len(test), test_start, gap)

    full_features = ds.features(FEATURE_SETS[FULL_SET])

    # 1) model selection on dev with walk-forward CV
    cv = {name: cross_validate(dev, full_features, name, cfg, gap) for name in cfg.models}
    ranked = sorted((m for m in cv if cv[m].get("folds")), key=lambda m: cv[m]["mae_mean"])
    best = ranked[0] if ranked else cfg.models[0]
    log.info("best model by CV MAE: %s", best)

    # 2) ablation for the best model
    ablation_cv = {
        fs: cross_validate(dev, ds.features(groups), best, cfg, gap) for fs, groups in FEATURE_SETS.items()
    }

    # 3) single evaluation on the held-out test period
    test_results: dict[str, Any] = {}
    test_preds = pd.DataFrame({"actual": test[TARGET]}, index=test.index)
    for fs, groups in FEATURE_SETS.items():
        feats = ds.features(groups)
        model, pred = _fit_predict(best, dev[feats], dev[TARGET], test[feats], cfg)
        test_results[fs] = regression_metrics(test[TARGET].to_numpy(), pred)
        test_preds[f"pred_{fs}"] = pred
        if fs == FULL_SET:
            best_model, best_pred = model, pred
    for name in cfg.models:
        if name == best:
            continue
        _, pred = _fit_predict(name, dev[full_features], dev[TARGET], test[full_features], cfg)
        test_results[f"{FULL_SET}__{name}"] = regression_metrics(test[TARGET].to_numpy(), pred)
    baselines = {
        name: regression_metrics(test[TARGET].to_numpy(), pred)
        for name, pred in baseline_predictions(dev, test, TARGET, ds.windows.pred).items()
    }

    w = ds.windows
    bt = {
        "cost_aware": backtest(test[TARGET], pd.Series(best_pred, index=test.index), w.pred, w.step, cfg.fee_pct),
        "zero_threshold": backtest(test[TARGET], pd.Series(best_pred, index=test.index), w.pred, w.step, cfg.fee_pct, 0.0),
        "zero_fee_zero_threshold": backtest(test[TARGET], pd.Series(best_pred, index=test.index), w.pred, w.step, 0.0, 0.0),
    }
    sanity = {
        "target_autocorrelation": autocorrelation(df[TARGET], lags=(1, 2, 3)),
        "permutation": permutation_test(best, dev[full_features], dev[TARGET], test[full_features], test[TARGET], cfg),
        "timestamp_audit": audit,
        "train_r2": regression_metrics(dev[TARGET].to_numpy(), best_model.predict(dev[full_features]))["r2"],
    }
    importance = feature_importance(best_model, full_features).head(15)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "windows": vars(ds.windows),
        "samples": {"total": len(df), "dev": len(dev), "test": len(test),
                    "dev_period": [str(dev.index.min()), str(dev.index.max())],
                    "test_period": [str(test.index.min()), str(test.index.max())]},
        "target_stats": df[TARGET].describe().to_dict(),
        "best_model": best,
        "cv_model_selection": cv,
        "cv_ablation": ablation_cv,
        "test": test_results,
        "test_baselines": baselines,
        "backtest": bt,
        "sanity": sanity,
        "feature_importance": importance.round(4).to_dict(),
        "_test_predictions": test_preds,
    }


def fit_final(ds: Dataset, model_name: str, cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ExperimentConfig()
    feats = ds.features(FEATURE_SETS[FULL_SET])
    model = make_model(model_name, cfg.random_state, cfg.noise).fit(ds.frame[feats], ds.frame[TARGET])
    return {
        "model": model,
        "model_name": model_name,
        "features": feats,
        "windows": vars(ds.windows),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_until": str(ds.frame.index.max()),
        "n_samples": len(ds.frame),
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Experiment report", "",
             f"Created {report['created_at']}. Windows (minutes): {report['windows']}.", "",
             f"Samples: {report['samples']}", "",
             f"**Best model (walk-forward CV MAE):** `{report['best_model']}`", "",
             "## Held-out test set", "",
             "| configuration | MAE | RMSE | R² | direction acc. | IC |", "|---|---|---|---|---|---|"]
    rows = {**report["test"], **{f"baseline: {k}": v for k, v in report["test_baselines"].items()}}
    for name, m in rows.items():
        lines.append(f"| {name} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['r2'])} | "
                     f"{_fmt(m['direction_accuracy'])} | {_fmt(m['information_coefficient'])} |")
    lines += ["", "## Walk-forward CV (development period)", "",
              "| model / feature set | folds | MAE | R² | direction acc. |", "|---|---|---|---|---|"]
    for title, block in (("model", report["cv_model_selection"]), ("ablation", report["cv_ablation"])):
        for name, m in block.items():
            if not m.get("folds"):
                continue
            lines.append(f"| {title}: {name} | {m['folds']} | {_fmt(m['mae_mean'])} ± {_fmt(m['mae_std'])} | "
                         f"{_fmt(m['r2_mean'])} ± {_fmt(m['r2_std'])} | {_fmt(m['direction_accuracy_mean'])} |")
    lines += ["", "## Backtest on the test set", "", "| variant | trades | total % | win rate | Sharpe | max DD % | buy&hold % |",
              "|---|---|---|---|---|---|---|"]
    for name, b in report["backtest"].items():
        lines.append(f"| {name} | {b['trades']} | {_fmt(b['total_return_pct'])} | {_fmt(b['win_rate'])} | "
                     f"{_fmt(b['sharpe_annualised'])} | {_fmt(b['max_drawdown_pct'])} | {_fmt(b['buy_and_hold_pct'])} |")
    s = report["sanity"]
    lines += ["", "## Sanity checks", "",
              f"- Target autocorrelation: {s['target_autocorrelation']}",
              f"- Permutation test (shuffled training target) R²: {_fmt(s['permutation']['r2_shuffled_mean'])}",
              f"- Train R² of best model: {_fmt(s['train_r2'])} (a value near 1 would signal memorisation)",
              f"- Timestamp audit: {s['timestamp_audit']}", "",
              "## Top features", ""]
    lines += [f"- {k}: {v}" for k, v in report["feature_importance"].items()]
    return "\n".join(lines) + "\n"


def save_outputs(report: dict[str, Any], final: dict[str, Any] | None, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    preds = report.get("_test_predictions")
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    paths["report_json"] = out_dir / "report.json"
    paths["report_json"].write_text(json.dumps(clean, indent=2, default=str))
    paths["report_md"] = out_dir / "report.md"
    paths["report_md"].write_text(render_markdown(clean))
    if preds is not None:
        paths["test_predictions"] = out_dir / "test_predictions.csv"
        preds.to_csv(paths["test_predictions"])
    if final is not None:
        paths["model"] = out_dir / "model.joblib"
        joblib.dump(final, paths["model"])
    return paths


def log_mlflow(report: dict[str, Any], paths: dict[str, Path], tracking_uri: str, experiment: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{report['best_model']}-{report['created_at'][:16]}"):
        mlflow.log_params({f"window_{k}": v for k, v in report["windows"].items()})
        mlflow.log_params({"best_model": report["best_model"], **{f"n_{k}": v for k, v in report["samples"].items() if isinstance(v, int)}})
        for fs, m in report["test"].items():
            for k, v in m.items():
                if isinstance(v, float) and not math.isnan(v):
                    mlflow.log_metric(f"test.{fs}.{k}", v)
        for k, v in report["backtest"]["cost_aware"].items():
            if isinstance(v, float) and not math.isnan(v):
                mlflow.log_metric(f"backtest.{k}", v)
        for p in paths.values():
            mlflow.log_artifact(str(p))
