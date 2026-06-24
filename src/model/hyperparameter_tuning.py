"""
Hyperparameter tuning via Optuna.
Optimizes LightGBM params on walk-forward validation MAE.

Usage:
    python -m src.model.hyperparameter_tuning
"""
import argparse
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import GLOBAL_SEED, PROJECT_ROOT


def load_tuning_config() -> dict:
    path = PROJECT_ROOT / "config" / "model_params.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["tuning"]


def objective(trial, X_train, y_train, X_val, y_val, base_params):
    """Optuna objective function."""
    config = load_tuning_config()
    space = config["search_space"]

    params = base_params.copy()
    params["num_leaves"] = trial.suggest_int("num_leaves", space["num_leaves"][0], space["num_leaves"][1])
    params["learning_rate"] = trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True)
    params["feature_fraction"] = trial.suggest_float("feature_fraction", space["feature_fraction"][0], space["feature_fraction"][1])
    params["bagging_fraction"] = trial.suggest_float("bagging_fraction", space["bagging_fraction"][0], space["bagging_fraction"][1])
    params["min_child_samples"] = trial.suggest_int("min_child_samples", space["min_child_samples"][0], space["min_child_samples"][1])
    params["reg_alpha"] = trial.suggest_float("reg_alpha", space["reg_alpha"][0], space["reg_alpha"][1])
    params["reg_lambda"] = trial.suggest_float("reg_lambda", space["reg_lambda"][0], space["reg_lambda"][1])

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    preds = model.predict(X_val)
    mae = np.mean(np.abs(y_val - preds))
    return mae


def run_tuning(X_train, y_train, X_val, y_val):
    """Run Optuna hyperparameter search."""
    config = load_tuning_config()

    path = PROJECT_ROOT / "config" / "model_params.yaml"
    with open(path) as f:
        full_config = yaml.safe_load(f)

    base_params = full_config["lightgbm"]["base"].copy()
    base_params.update(full_config["lightgbm"]["median"])

    study = optuna.create_study(
        study_name=config["study_name"],
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, base_params),
        n_trials=config["n_trials"],
        timeout=config.get("timeout_seconds", 3600),
    )

    logger.info(f"Best MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params, study.best_value


def main():
    from src.model.train import prepare_training_data, load_feature_columns

    df, feature_cols = prepare_training_data()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    cutoff = df["game_date"].max() - pd.Timedelta(weeks=4)

    train = df[df["game_date"] <= cutoff]
    val = df[df["game_date"] > cutoff]

    X_train = train[feature_cols].values
    y_train = train["actual_strikeouts"].values
    X_val = val[feature_cols].values
    y_val = val["actual_strikeouts"].values

    best_params, best_mae = run_tuning(X_train, y_train, X_val, y_val)

    # Save best params
    out_path = PROJECT_ROOT / "config" / "tuned_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(best_params, f)
    logger.info(f"Saved tuned params to {out_path}")


if __name__ == "__main__":
    main()
