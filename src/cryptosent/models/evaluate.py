"""Metrics and a fee-aware trading backtest (report 5.4.4 "missing practical
validation")."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    moving = y_true != 0
    direction = float(np.mean(np.sign(y_pred[moving]) == np.sign(y_true[moving]))) if moving.any() else float("nan")
    if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        ic = float(spearmanr(y_true, y_pred).statistic)
    else:
        ic = 0.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "direction_accuracy": direction,
        "information_coefficient": ic,
        "n": int(len(y_true)),
    }


def backtest(
    y_true: pd.Series,
    y_pred: pd.Series,
    pred_min: int,
    step_min: int,
    fee_pct: float = 0.1,
    threshold_pct: float | None = None,
) -> dict[str, float]:
    """Trade the prediction window: long if pred > threshold, short if
    pred < -threshold, flat otherwise. Each trade pays ``fee_pct`` per side.

    Only non-overlapping trades are counted (every ``ceil(pred/step)``-th sample).
    The default threshold is the round-trip fee: only trade when the model
    expects to beat costs."""
    if threshold_pct is None:
        threshold_pct = 2 * fee_pct
    every = max(1, math.ceil(pred_min / step_min))
    yt = y_true.iloc[::every].to_numpy(dtype=float)
    yp = y_pred.iloc[::every].to_numpy(dtype=float)
    pos = np.where(yp > threshold_pct, 1.0, np.where(yp < -threshold_pct, -1.0, 0.0))
    pnl = pos * yt - np.abs(pos) * 2 * fee_pct
    trades = int(np.count_nonzero(pos))
    equity = np.cumsum(pnl)
    drawdown = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity)) if len(equity) else 0.0
    periods_per_year = 365 * 24 * 60 / (every * step_min)
    sharpe = float(pnl.mean() / pnl.std() * math.sqrt(periods_per_year)) if pnl.std() > 0 else 0.0
    traded = pnl[pos != 0]
    return {
        "threshold_pct": float(threshold_pct),
        "fee_pct_per_side": float(fee_pct),
        "periods": int(len(pnl)),
        "trades": trades,
        "total_return_pct": float(pnl.sum()),
        "mean_return_per_trade_pct": float(traded.mean()) if trades else 0.0,
        "win_rate": float(np.mean(traded > 0)) if trades else float("nan"),
        "sharpe_annualised": sharpe,
        "max_drawdown_pct": drawdown,
        "buy_and_hold_pct": float(yt.sum()),
    }
