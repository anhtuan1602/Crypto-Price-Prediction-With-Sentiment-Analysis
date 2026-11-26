"""Model zoo: the report's learners plus LightGBM, each wrapped in a pipeline
whose preprocessing (correlation filter, scaling, noise) is fitted on the
training fold only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of each feature pair with |Pearson r| > threshold (report 4.1.2a),
    learnt on training data only."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr)
        keep = []
        for j in range(X.shape[1]):
            if np.nanstd(X[:, j]) == 0:
                continue  # constant columns carry nothing
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep or [0], dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class NoisyRegressor(BaseEstimator, RegressorMixin):
    """Gaussian noise injection on training features only (report 4.1.2b):
    x + N(0, noise * std(x)). Prediction uses clean inputs."""

    def __init__(self, estimator=None, noise: float = 0.05, random_state: int = 42):
        self.estimator = estimator
        self.noise = noise
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        if self.noise > 0:
            X = X + rng.normal(0, 1, X.shape) * (self.noise * X.std(axis=0, keepdims=True))
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))

    @property
    def feature_importances_(self):
        return getattr(self.estimator_, "feature_importances_", None)


def _pipeline(estimator, noise: float, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("corr", CorrelationFilter(0.95)),
            ("scale", MinMaxScaler()),
            ("model", NoisyRegressor(estimator, noise=noise, random_state=random_state)),
        ]
    )


def make_model(name: str, random_state: int = 42, noise: float = 0.05) -> Pipeline:
    if name == "linear":
        est = LinearRegression()
    elif name == "ridge":
        est = Ridge(alpha=10.0)
    elif name == "random_forest":
        est = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=20,
            max_features="sqrt", n_jobs=-1, random_state=random_state,
        )
    elif name == "gradient_boosting":
        est = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
            min_samples_leaf=30, random_state=random_state,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor

        est = LGBMRegressor(
            n_estimators=400, learning_rate=0.02, num_leaves=15, max_depth=4,
            min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=5.0, random_state=random_state, verbose=-1, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model {name}")
    return _pipeline(est, noise if name not in ("linear", "ridge") else 0.0, random_state)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401  (needs libgomp at runtime)

        return True
    except (ImportError, OSError):
        return False


MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting"] + (["lightgbm"] if _lightgbm_available() else [])


def feature_importance(model: Pipeline, feature_names: list[str]) -> pd.Series:
    kept = np.array(feature_names)[model.named_steps["corr"].keep_]
    inner = model.named_steps["model"].estimator_
    if hasattr(inner, "feature_importances_"):
        values = np.asarray(inner.feature_importances_, dtype=float)
    elif hasattr(inner, "coef_"):
        values = np.abs(np.ravel(inner.coef_))
    else:
        return pd.Series(dtype=float)
    if values.sum() > 0:
        values = values / values.sum()
    return pd.Series(values, index=kept).sort_values(ascending=False)
