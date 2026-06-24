"""
Inference pipeline.
Loads production models, runs prediction for a date's starters.

Usage:
    python -m src.model.predict --date 2024-08-01
"""
import argparse
import json
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PRODUCTION_MODEL_DIR, PROJECT_ROOT, GLOBAL_SEED


def load_feature_columns() -> list[str]:
    path = PROJECT_ROOT / "config" / "features.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["feature_columns"]


def find_latest_model(model_dir: Path = None) -> dict:
    """Find the most recent model artifacts in the given directory."""
    if model_dir is None:
        model_dir = PRODUCTION_MODEL_DIR

    # Check for metadata files
    meta_files = sorted(model_dir.glob("metadata_*.json"), reverse=True)
    if not meta_files:
        # Fall back to experiments
        from config.settings import EXPERIMENT_MODEL_DIR
        meta_files = sorted(EXPERIMENT_MODEL_DIR.glob("metadata_*.json"), reverse=True)

    if not meta_files:
        raise FileNotFoundError("No trained models found. Run training first.")

    with open(meta_files[0]) as f:
        metadata = json.load(f)

    return metadata


def load_models(metadata: dict) -> tuple:
    """Load LightGBM models and stacking model from metadata paths."""
    models = {}
    for name in ["median", "lower", "upper"]:
        path = metadata["models"][name]
        models[name] = lgb.Booster(model_file=path)

    stacking = joblib.load(metadata["models"]["stacking"])
    return models, stacking


def predict(feature_matrix: pd.DataFrame,
            models: dict = None, stacking_model=None,
            metadata: dict = None) -> pd.DataFrame:
    """
    Generate predictions for a feature matrix.

    Returns DataFrame with: pitcher_id, game_date, game_pk,
    predicted_strikeouts, pred_ci_lower, pred_ci_upper
    """
    feature_cols = load_feature_columns()

    if models is None or stacking_model is None:
        if metadata is None:
            metadata = find_latest_model()
        models, stacking_model = load_models(metadata)

    # Ensure all features exist
    X = feature_matrix.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    X_features = X[feature_cols]

    # LightGBM predictions
    median_preds = models["median"].predict(X_features)
    lower_preds = models["lower"].predict(X_features)
    upper_preds = models["upper"].predict(X_features)

    # Baseline for stacking
    baseline = (X.get("k_per_9_rolling", 8.5) / 9) * X.get("ip_per_start", 5.3)
    if isinstance(baseline, pd.Series):
        baseline = baseline.fillna(5.0).values
    else:
        baseline = np.full(len(X), 5.0)

    days_rest = X.get("days_rest", pd.Series(5.0, index=X.index)).fillna(5.0).values

    # Stacking
    X_stack = np.column_stack([median_preds, baseline, days_rest])
    stacked_preds = stacking_model.predict(X_stack)

    # Clamp predictions to reasonable range
    stacked_preds = np.clip(stacked_preds, 0, 20)
    lower_preds = np.clip(lower_preds, 0, 20)
    upper_preds = np.clip(upper_preds, 0, 20)

    # Ensure lower <= median <= upper
    lower_preds = np.minimum(lower_preds, stacked_preds)
    upper_preds = np.maximum(upper_preds, stacked_preds)

    result = pd.DataFrame({
        "pitcher_id": feature_matrix.get("pitcher_id", 0),
        "game_date": feature_matrix.get("game_date", ""),
        "game_pk": feature_matrix.get("game_pk", 0),
        "predicted_strikeouts": np.round(stacked_preds, 2),
        "pred_ci_lower": np.round(lower_preds, 2),
        "pred_ci_upper": np.round(upper_preds, 2),
    })

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate K predictions")
    parser.add_argument("--date", type=str, required=True, help="Prediction date YYYY-MM-DD")
    args = parser.parse_args()

    # Load features
    from config.settings import FEATURES_DIR
    feat_path = FEATURES_DIR / "pitcher_features" / f"{args.date}.parquet"

    if not feat_path.exists():
        logger.error(f"No features found for {args.date}. Run feature pipeline first.")
        return

    features = pd.read_parquet(feat_path)
    predictions = predict(features)

    logger.info(f"Predictions for {args.date}:")
    logger.info(f"\n{predictions.to_string(index=False)}")


if __name__ == "__main__":
    main()
