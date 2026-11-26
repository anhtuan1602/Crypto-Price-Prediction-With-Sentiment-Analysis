"""Naive baselines (report table 7.6), all using only information available
at decision time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def last_return(frame: pd.DataFrame, pred_min: int) -> pd.Series:
    """Return over the last ``pred_min`` minutes, known at t. Uses the
    technical return columns when the horizon matches, else 10-minute return."""
    col = f"return_{pred_min}" if f"return_{pred_min}" in frame else "return_10"
    return frame[col].fillna(0.0)


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str, pred_min: int) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    preds["zero_no_change"] = np.zeros(len(test))
    preds["train_mean"] = np.full(len(test), train[target].mean())
    x_tr = last_return(train, pred_min).to_numpy().reshape(-1, 1)
    x_te = last_return(test, pred_min).to_numpy().reshape(-1, 1)
    ar = LinearRegression().fit(x_tr, train[target].to_numpy())
    preds["ar1_last_return"] = ar.predict(x_te)
    preds["momentum_sign"] = np.sign(x_te.ravel()) * np.abs(train[target]).mean()
    return preds
