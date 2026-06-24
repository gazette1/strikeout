"""
Walk-forward time-series cross-validation.
Each fold advances by 1 week, training on all prior data.

Usage:
    python -m src.evaluation.walk_forward
"""
import argparse
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import GLOBAL_SEED, PROJECT_ROOT, PREDICTIONS_DIR
from src.evaluation.metrics import full_evaluation_report


def load_model_params():
    path = PROJECT_ROOT / "config" / "model_params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_feature_columns():
    path = PROJECT_ROOT / "config" / "features.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["feature_columns"]


def walk_forward_backtest(df: pd.DataFrame,
                          feature_cols: list[str],
                          target: str = "actual_strikeouts",
                          min_train_weeks: int = 8,
                          val_window_weeks: int = 1,
                          seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """
    Walk-forward validation where each fold advances by 1 week.
    Returns DataFrame of all out-of-sample predictions with actuals.
    """
    config = load_model_params()
    lgbm_base = config["lightgbm"]["base"]
    median_params = {**lgbm_base, **config["lightgbm"]["median"]}
    lower_params = {**lgbm_base, **config["lightgbm"]["lower"]}
    upper_params = {**lgbm_base, **config["lightgbm"]["upper"]}
    training_cfg = config["training"]

    df = df.sort_values("game_date").reset_index(drop=True)
    df["game_date_dt"] = pd.to_datetime(df["game_date"])

    # Assign week numbers (continuous across years)
    df["year"] = df["game_date_dt"].dt.year
    df["week_of_year"] = df["game_date_dt"].dt.isocalendar().week.astype(int)
    min_year = df["year"].min()
    df["week_num"] = (df["year"] - min_year) * 52 + df["week_of_year"]

    weeks = sorted(df["week_num"].unique())

    all_predictions = []

    for i in range(min_train_weeks, len(weeks) - val_window_weeks + 1):
        train_weeks = set(weeks[:i])
        val_weeks = set(weeks[i:i + val_window_weeks])

        train_mask = df["week_num"].isin(train_weeks)
        val_mask = df["week_num"].isin(val_weeks)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, target]

        if len(X_val) == 0:
            continue

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval_ds = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train median model
        model_med = lgb.train(
            params=median_params,
            train_set=dtrain,
            valid_sets=[dval_ds],
            num_boost_round=training_cfg["num_boost_round"],
            callbacks=[lgb.early_stopping(training_cfg["early_stopping_rounds"]),
                      lgb.log_evaluation(0)],
        )

        # Train quantile models
        model_lower = lgb.train(
            params=lower_params, train_set=dtrain, valid_sets=[dval_ds],
            num_boost_round=training_cfg["num_boost_round"],
            callbacks=[lgb.early_stopping(training_cfg["early_stopping_rounds"]),
                      lgb.log_evaluation(0)],
        )
        model_upper = lgb.train(
            params=upper_params, train_set=dtrain, valid_sets=[dval_ds],
            num_boost_round=training_cfg["num_boost_round"],
            callbacks=[lgb.early_stopping(training_cfg["early_stopping_rounds"]),
                      lgb.log_evaluation(0)],
        )

        preds_med = model_med.predict(X_val)
        preds_lower = model_lower.predict(X_val)
        preds_upper = model_upper.predict(X_val)

        fold_results = df.loc[val_mask, ["pitcher_id", "game_date", target]].copy()
        fold_results["predicted_k"] = preds_med
        fold_results["pred_lower"] = preds_lower
        fold_results["pred_upper"] = preds_upper
        fold_results["fold"] = i
        all_predictions.append(fold_results)

        if i % 10 == 0:
            fold_mae = np.mean(np.abs(y_val.values - preds_med))
            logger.info(f"Fold {i}/{len(weeks)}: val_size={len(X_val)}, MAE={fold_mae:.3f}")

    return pd.concat(all_predictions, ignore_index=True)


def main():
    from src.model.train import prepare_training_data

    df, feature_cols = prepare_training_data()

    logger.info(f"Running walk-forward backtest on {len(df)} samples")
    results = walk_forward_backtest(df, feature_cols)

    # Evaluate
    report = full_evaluation_report(
        results["actual_strikeouts"].values,
        results["predicted_k"].values,
        results.get("pred_lower", pd.Series()).values if "pred_lower" in results.columns else None,
        results.get("pred_upper", pd.Series()).values if "pred_upper" in results.columns else None,
    )

    logger.info("=== Walk-Forward Backtest Results ===")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    # Save results
    out_dir = PREDICTIONS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_dir / "walk_forward_results.parquet", index=False)

    import json
    with open(out_dir / "walk_forward_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
