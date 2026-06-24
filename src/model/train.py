"""
Model training pipeline.
Trains LightGBM median + quantile models, then a Ridge stacking layer.

Usage:
    python -m src.model.train [--data-path path] [--output-dir path]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
import joblib
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    FEATURES_DIR, PRODUCTION_MODEL_DIR, EXPERIMENT_MODEL_DIR,
    MODEL_REGISTRY, GLOBAL_SEED, PROJECT_ROOT,
)


def load_model_params() -> dict:
    """Load model hyperparameters from YAML config."""
    path = PROJECT_ROOT / "config" / "model_params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_feature_columns() -> list[str]:
    """Load ordered feature column names from features.yaml."""
    path = PROJECT_ROOT / "config" / "features.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["feature_columns"]


def prepare_training_data(data_path: Path = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and prepare training data from feature matrices.
    Returns (DataFrame, feature_column_list).
    """
    feature_cols = load_feature_columns()

    if data_path and data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        # Load all feature files
        feat_dir = FEATURES_DIR / "pitcher_features"
        if not feat_dir.exists():
            raise FileNotFoundError(f"No feature data found at {feat_dir}")

        frames = [pd.read_parquet(f) for f in sorted(feat_dir.glob("*.parquet"))]
        if not frames:
            raise ValueError("No feature files found")
        df = pd.concat(frames, ignore_index=True)

    # Require target
    if "actual_strikeouts" not in df.columns:
        # Try to join from game logs
        from config.settings import STAGING_GAMES
        if STAGING_GAMES.exists():
            games = pd.read_parquet(STAGING_GAMES)
            df = df.merge(
                games[["game_pk", "pitcher_id", "strikeouts"]].rename(
                    columns={"strikeouts": "actual_strikeouts"}
                ),
                on=["game_pk", "pitcher_id"],
                how="left"
            )

    # Drop rows without target
    df = df.dropna(subset=["actual_strikeouts"])

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df, feature_cols


def train_lgbm_model(X_train, y_train, X_val, y_val, params: dict,
                     training_config: dict) -> lgb.Booster:
    """Train a single LightGBM model."""
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(training_config.get("early_stopping_rounds", 50)),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=training_config.get("num_boost_round", 1000),
        callbacks=callbacks,
    )

    return model


def train_stacking_layer(lgbm_preds: np.ndarray, baseline_preds: np.ndarray,
                          days_rest: np.ndarray, y_true: np.ndarray,
                          alpha: float = 1.0) -> Ridge:
    """
    Train Ridge regression stacking layer.
    Inputs: LightGBM prediction, season K/9 × IP baseline, days rest.
    """
    X_stack = np.column_stack([lgbm_preds, baseline_preds, days_rest])
    ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=GLOBAL_SEED)
    ridge.fit(X_stack, y_true)
    return ridge


def train_full_pipeline(data_path: Path = None, output_dir: Path = None):
    """
    Full training pipeline:
    1. Load data
    2. Time-based train/val split (last 4 weeks = validation)
    3. Train median, lower, upper LightGBM models
    4. Train stacking layer
    5. Save all artifacts
    """
    config = load_model_params()
    lgbm_base = config["lightgbm"]["base"]
    training_cfg = config["training"]
    stacking_cfg = config["stacking"]

    df, feature_cols = prepare_training_data(data_path)
    logger.info(f"Training data: {len(df)} rows, {len(feature_cols)} features")

    # Time-based split: last 4 weeks = validation
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    cutoff = df["game_date"].max() - pd.Timedelta(weeks=4)

    train = df[df["game_date"] <= cutoff]
    val = df[df["game_date"] > cutoff]

    logger.info(f"Train: {len(train)} rows, Val: {len(val)} rows")

    X_train = train[feature_cols]
    y_train = train["actual_strikeouts"]
    X_val = val[feature_cols]
    y_val = val["actual_strikeouts"]

    # Train 3 quantile models
    models = {}
    for name in ["median", "lower", "upper"]:
        quantile_params = config["lightgbm"][name]
        params = {**lgbm_base, **quantile_params}
        logger.info(f"Training {name} model (alpha={params.get('alpha', 'N/A')})")
        models[name] = train_lgbm_model(X_train, y_train, X_val, y_val, params, training_cfg)

    # Generate predictions for stacking
    lgbm_val_preds = models["median"].predict(X_val)

    # Baseline: season K/9 × IP per start (from features)
    baseline_val = (val.get("k_per_9_rolling", 8.5) / 9) * val.get("ip_per_start", 5.3)
    if isinstance(baseline_val, pd.Series):
        baseline_val = baseline_val.fillna(5.0).values
    else:
        baseline_val = np.full(len(val), 5.0)

    days_rest_val = val.get("days_rest", pd.Series(5.0, index=val.index)).fillna(5.0).values

    # Train stacking layer
    stacking_model = train_stacking_layer(
        lgbm_val_preds, baseline_val, days_rest_val,
        y_val.values, alpha=stacking_cfg["alpha"]
    )

    # Evaluate
    from src.evaluation.metrics import compute_mae, compute_rmse

    # Stacked predictions
    X_stack_val = np.column_stack([lgbm_val_preds, baseline_val, days_rest_val])
    stacked_preds = stacking_model.predict(X_stack_val)

    mae = compute_mae(y_val.values, stacked_preds)
    rmse = compute_rmse(y_val.values, stacked_preds)
    logger.info(f"Validation MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    # Save models
    if output_dir is None:
        output_dir = EXPERIMENT_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")

    for name, model in models.items():
        model_path = output_dir / f"lgbm_{name}_{timestamp}.txt"
        model.save_model(str(model_path))
        logger.info(f"Saved {name} model to {model_path}")

    stacking_path = output_dir / f"stacking_{timestamp}.pkl"
    joblib.dump(stacking_model, stacking_path)
    logger.info(f"Saved stacking model to {stacking_path}")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "train_rows": len(train),
        "val_rows": len(val),
        "val_mae": round(mae, 4),
        "val_rmse": round(rmse, 4),
        "feature_columns": feature_cols,
        "models": {
            "median": str(output_dir / f"lgbm_median_{timestamp}.txt"),
            "lower": str(output_dir / f"lgbm_lower_{timestamp}.txt"),
            "upper": str(output_dir / f"lgbm_upper_{timestamp}.txt"),
            "stacking": str(stacking_path),
        },
    }

    meta_path = output_dir / f"metadata_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return models, stacking_model, metadata


def main():
    parser = argparse.ArgumentParser(description="Train K-Predictor models")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    train_full_pipeline(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
