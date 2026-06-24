"""
Tests for model training and prediction.
Uses lightweight in-memory training to avoid dependency on real data files.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Helper: quick LightGBM train (10 rounds) returning model + predictions
# ─────────────────────────────────────────────────────────────────────────────
def _quick_lgbm_train(X_train, y_train, n_rounds: int = 10):
    """Train a minimal LightGBM model without callbacks for speed."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=n_rounds)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. test_lightgbm_train
# ─────────────────────────────────────────────────────────────────────────────
def test_lightgbm_train(sample_feature_matrix):
    """
    Train a quick (10-round) LightGBM model and verify it produces
    numeric predictions of the correct shape.
    """
    import lightgbm as lgb

    df = sample_feature_matrix.copy()
    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    model = _quick_lgbm_train(X, y, n_rounds=10)

    assert isinstance(model, lgb.Booster)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(df)
    assert not np.any(np.isnan(preds)), "LightGBM predictions should not contain NaN"


def test_lightgbm_train_with_nan_features(sample_feature_matrix):
    """
    LightGBM should handle NaN feature values natively without raising.
    """
    df = sample_feature_matrix.copy()
    # Introduce NaN in some native-NaN columns
    df.loc[0:5, "ump_k_boost"] = np.nan
    df.loc[0:5, "catcher_framing_runs"] = np.nan

    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    model = _quick_lgbm_train(X, y, n_rounds=10)
    preds = model.predict(X)
    assert len(preds) == len(df)


# ─────────────────────────────────────────────────────────────────────────────
# 2. test_stacking_layer
# ─────────────────────────────────────────────────────────────────────────────
def test_stacking_layer(sample_feature_matrix):
    """
    Ridge stacking layer accepts LightGBM predictions + baseline + days_rest
    and produces an output array of the correct shape.
    """
    from src.model.train import train_stacking_layer
    from sklearn.linear_model import Ridge

    df = sample_feature_matrix.copy()
    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    lgbm_model = _quick_lgbm_train(X, y, n_rounds=10)
    lgbm_preds = lgbm_model.predict(X)

    baseline = (df["k_per_9_rolling"] / 9) * df["ip_per_start"]
    days_rest = df["days_rest"].fillna(5.0).values

    stacking = train_stacking_layer(lgbm_preds, baseline.values, days_rest, y)

    assert isinstance(stacking, Ridge)

    # Stack inputs
    X_stack = np.column_stack([lgbm_preds, baseline.values, days_rest])
    stack_preds = stacking.predict(X_stack)

    assert isinstance(stack_preds, np.ndarray)
    assert len(stack_preds) == len(df)
    assert not np.any(np.isnan(stack_preds))


def test_stacking_layer_coefficients():
    """
    Ridge stacking model has 3 features (lgbm, baseline, days_rest).
    """
    from src.model.train import train_stacking_layer

    rng = np.random.default_rng(42)
    n = 50
    lgbm_preds = rng.uniform(3, 10, n)
    baseline = rng.uniform(3, 9, n)
    days_rest = rng.uniform(4, 8, n)
    y_true = rng.uniform(2, 12, n)

    stacking = train_stacking_layer(lgbm_preds, baseline, days_rest, y_true)
    # Ridge has n_features_in_ == 3
    assert stacking.n_features_in_ == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. test_predict_output_shape
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_output_shape(sample_feature_matrix):
    """
    predict() called with inline models returns a DataFrame with
    the expected columns and one row per input row.
    """
    import lightgbm as lgb
    from src.model.predict import predict
    from src.model.train import train_stacking_layer

    df = sample_feature_matrix.copy()
    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    # Train three models (median, lower, upper) inline
    med_model = _quick_lgbm_train(X, y, n_rounds=10)

    params_lower = {
        "objective": "quantile", "alpha": 0.1, "metric": "quantile",
        "num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 42,
    }
    params_upper = {
        "objective": "quantile", "alpha": 0.9, "metric": "quantile",
        "num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X, label=y)
    lower_model = lgb.train(params_lower, dtrain, num_boost_round=10)
    upper_model = lgb.train(params_upper, dtrain, num_boost_round=10)

    models = {"median": med_model, "lower": lower_model, "upper": upper_model}

    # Train a stacking model
    lgbm_preds = med_model.predict(X)
    baseline = ((df["k_per_9_rolling"] / 9) * df["ip_per_start"]).fillna(5.0).values
    days_rest = df["days_rest"].fillna(5.0).values
    stacking = train_stacking_layer(lgbm_preds, baseline, days_rest, y)

    result = predict(df, models=models, stacking_model=stacking)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)

    expected_cols = {"predicted_strikeouts", "pred_ci_lower", "pred_ci_upper"}
    missing = expected_cols - set(result.columns)
    assert not missing, f"Missing output columns: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. test_prediction_range
# ─────────────────────────────────────────────────────────────────────────────
def test_prediction_range(sample_feature_matrix):
    """
    All median predictions should fall within the plausible range [0, 20]
    (no pitcher throws 20+ Ks in a realistic model).
    """
    import lightgbm as lgb
    from src.model.predict import predict
    from src.model.train import train_stacking_layer

    df = sample_feature_matrix.copy()
    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    med_model = _quick_lgbm_train(X, y, n_rounds=10)

    params_q = {
        "objective": "quantile", "alpha": 0.5, "metric": "quantile",
        "num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X, label=y)
    q_model = lgb.train(params_q, dtrain, num_boost_round=10)

    models = {"median": med_model, "lower": q_model, "upper": q_model}
    lgbm_preds = med_model.predict(X)
    baseline = ((df["k_per_9_rolling"] / 9) * df["ip_per_start"]).fillna(5.0).values
    days_rest = df["days_rest"].fillna(5.0).values
    stacking = train_stacking_layer(lgbm_preds, baseline, days_rest, y)

    result = predict(df, models=models, stacking_model=stacking)

    assert (result["predicted_strikeouts"] >= 0).all(), (
        "Some predictions are negative"
    )
    assert (result["predicted_strikeouts"] <= 20).all(), (
        f"Some predictions exceed 20: {result['predicted_strikeouts'].max()}"
    )


def test_prediction_range_direct():
    """
    Directly verify LightGBM predictions on realistic feature values
    stay within [0, 20] after 10 training rounds on this dataset.
    """
    rng = np.random.default_rng(42)
    n = 100
    X = rng.uniform(0, 1, (n, 10))
    y = rng.uniform(0, 12, n)  # Realistic strikeout range

    model = _quick_lgbm_train(X, y, n_rounds=10)
    preds = model.predict(X)

    assert np.all(preds >= 0), "Predictions should be >= 0"
    assert np.all(preds <= 20), "Predictions should be <= 20 for this range"


# ─────────────────────────────────────────────────────────────────────────────
# 5. test_confidence_interval_ordering
# ─────────────────────────────────────────────────────────────────────────────
def test_confidence_interval_ordering(sample_feature_matrix):
    """
    For all rows: pred_ci_lower <= predicted_strikeouts <= pred_ci_upper.
    """
    import lightgbm as lgb
    from src.model.predict import predict
    from src.model.train import train_stacking_layer

    df = sample_feature_matrix.copy()
    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    med_model = _quick_lgbm_train(X, y, n_rounds=10)

    params_lower = {
        "objective": "quantile", "alpha": 0.1, "metric": "quantile",
        "num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 42,
    }
    params_upper = {
        "objective": "quantile", "alpha": 0.9, "metric": "quantile",
        "num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X, label=y)
    lower_model = lgb.train(params_lower, dtrain, num_boost_round=10)
    upper_model = lgb.train(params_upper, dtrain, num_boost_round=10)

    models = {"median": med_model, "lower": lower_model, "upper": upper_model}
    lgbm_preds = med_model.predict(X)
    baseline = ((df["k_per_9_rolling"] / 9) * df["ip_per_start"]).fillna(5.0).values
    days_rest = df["days_rest"].fillna(5.0).values
    stacking = train_stacking_layer(lgbm_preds, baseline, days_rest, y)

    result = predict(df, models=models, stacking_model=stacking)

    # lower <= median for every row
    assert (result["pred_ci_lower"] <= result["predicted_strikeouts"] + 1e-6).all(), (
        "Some lower CI values exceed the median prediction"
    )
    # median <= upper for every row
    assert (result["predicted_strikeouts"] <= result["pred_ci_upper"] + 1e-6).all(), (
        "Some median predictions exceed the upper CI"
    )


def test_confidence_interval_widths_positive(sample_feature_matrix):
    """
    CI width (upper - lower) should be non-negative for all rows.
    """
    import lightgbm as lgb
    from src.model.predict import predict
    from src.model.train import train_stacking_layer

    df = sample_feature_matrix.copy()
    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    params_lower = {"objective": "quantile", "alpha": 0.1, "metric": "quantile",
                    "num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 42}
    params_upper = {"objective": "quantile", "alpha": 0.9, "metric": "quantile",
                    "num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 42}
    dtrain = lgb.Dataset(X, label=y)
    med_model = _quick_lgbm_train(X, y)
    lower_model = lgb.train(params_lower, dtrain, num_boost_round=10)
    upper_model = lgb.train(params_upper, dtrain, num_boost_round=10)

    models = {"median": med_model, "lower": lower_model, "upper": upper_model}
    lgbm_preds = med_model.predict(X)
    baseline = ((df["k_per_9_rolling"] / 9) * df["ip_per_start"]).fillna(5.0).values
    days_rest = df["days_rest"].fillna(5.0).values
    stacking = train_stacking_layer(lgbm_preds, baseline, days_rest, y)

    result = predict(df, models=models, stacking_model=stacking)
    ci_width = result["pred_ci_upper"] - result["pred_ci_lower"]
    assert (ci_width >= -1e-6).all(), "CI width should be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# 6. test_metrics_computation
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("y_true,y_pred,expected_mae", [
    ([5, 6, 7, 8], [5, 6, 7, 8], 0.0),        # Perfect predictions
    ([0, 10], [10, 0], 10.0),                   # Worst case: swapped
    ([5, 5, 5, 5], [6, 6, 6, 6], 1.0),         # Constant offset
])
def test_compute_mae(y_true, y_pred, expected_mae):
    """compute_mae returns the correct mean absolute error."""
    from src.evaluation.metrics import compute_mae

    result = compute_mae(np.array(y_true, dtype=float), np.array(y_pred, dtype=float))
    assert result == pytest.approx(expected_mae, abs=1e-6)


@pytest.mark.parametrize("y_true,y_pred,expected_rmse", [
    ([3, 4], [3, 4], 0.0),                      # Perfect
    ([0, 0], [3, 4], pytest.approx(3.5355, abs=0.001)),  # sqrt(25/2)
])
def test_compute_rmse(y_true, y_pred, expected_rmse):
    """compute_rmse returns the correct root mean squared error."""
    from src.evaluation.metrics import compute_rmse

    result = compute_rmse(np.array(y_true, dtype=float), np.array(y_pred, dtype=float))
    assert result == pytest.approx(expected_rmse, abs=0.01)


def test_full_evaluation_report_keys():
    """full_evaluation_report returns a dict with all expected keys."""
    from src.evaluation.metrics import full_evaluation_report

    y_true = np.array([5, 6, 7, 8, 9, 10], dtype=float)
    y_pred = np.array([5.1, 6.2, 6.8, 8.3, 8.9, 10.1], dtype=float)

    report = full_evaluation_report(y_true, y_pred)

    assert "mae" in report
    assert "rmse" in report
    assert "n_samples" in report
    assert "over_under_accuracy" in report
    assert "mae_by_tier" in report
    assert report["n_samples"] == 6
    assert report["mae"] >= 0
    assert report["rmse"] >= report["mae"]  # RMSE >= MAE always


def test_full_evaluation_report_with_intervals():
    """full_evaluation_report includes interval calibration when CI provided."""
    from src.evaluation.metrics import full_evaluation_report

    y_true = np.array([5, 6, 7, 8], dtype=float)
    y_pred = np.array([5, 6, 7, 8], dtype=float)
    lower = y_pred - 2.0
    upper = y_pred + 2.0

    report = full_evaluation_report(y_true, y_pred, lower=lower, upper=upper)

    assert "interval_calibration_90" in report
    assert 0.0 <= report["interval_calibration_90"] <= 1.0
    # All actuals within ±2 of perfect preds → 100% calibration
    assert report["interval_calibration_90"] == pytest.approx(1.0)


def test_metrics_perfect_predictions():
    """All metrics should be 0 for perfect predictions."""
    from src.evaluation.metrics import compute_mae, compute_rmse

    y = np.array([5, 6, 7, 8, 9, 10], dtype=float)
    assert compute_mae(y, y) == pytest.approx(0.0)
    assert compute_rmse(y, y) == pytest.approx(0.0)
