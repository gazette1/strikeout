"""
Quantile model utilities.
Provides helpers for training and evaluating quantile regression models.
"""
import lightgbm as lgb
import numpy as np
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROJECT_ROOT


def get_quantile_params(quantile: float) -> dict:
    """Build LightGBM params for a specific quantile."""
    path = PROJECT_ROOT / "config" / "model_params.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)

    base = config["lightgbm"]["base"].copy()
    base["objective"] = "quantile"
    base["alpha"] = quantile
    base["metric"] = "quantile"
    return base


def train_quantile_model(X_train, y_train, X_val, y_val,
                          quantile: float = 0.5,
                          num_boost_round: int = 1000,
                          early_stopping_rounds: int = 50) -> lgb.Booster:
    """Train a quantile regression LightGBM model."""
    params = get_quantile_params(quantile)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )

    return model


def evaluate_interval_calibration(y_true: np.ndarray,
                                   lower: np.ndarray,
                                   upper: np.ndarray) -> float:
    """
    Compute the fraction of actuals within the prediction interval.
    Should be approximately 90% for a 90% CI (5th to 95th percentile).
    """
    in_interval = ((y_true >= lower) & (y_true <= upper)).mean()
    return float(in_interval)
