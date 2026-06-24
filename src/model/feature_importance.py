"""
Feature importance analysis using SHAP, native gain, and permutation importance.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def shap_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP feature importance using TreeExplainer."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        importance = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "feature": X.columns,
            "shap_importance": importance,
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return pd.DataFrame()


def native_gain_importance(model) -> pd.DataFrame:
    """LightGBM native feature importance (gain-based)."""
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    return pd.DataFrame({
        "feature": names,
        "gain_importance": importance,
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray,
                           n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Permutation importance: shuffle each feature and measure MAE increase.
    """
    from config.settings import GLOBAL_SEED
    rng = np.random.RandomState(seed or GLOBAL_SEED)

    base_preds = model.predict(X)
    base_mae = np.mean(np.abs(y - base_preds))

    results = []
    for col in X.columns:
        maes = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)
            preds = model.predict(X_shuffled)
            maes.append(np.mean(np.abs(y - preds)))

        mean_mae = np.mean(maes)
        results.append({
            "feature": col,
            "permutation_importance": mean_mae - base_mae,
        })

    return pd.DataFrame(results).sort_values(
        "permutation_importance", ascending=False
    ).reset_index(drop=True)


def full_importance_report(model, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Run all three importance methods and merge into a single report."""
    shap_df = shap_importance(model, X)
    gain_df = native_gain_importance(model)
    perm_df = permutation_importance(model, X, y)

    report = gain_df.merge(perm_df, on="feature", how="outer")
    if not shap_df.empty:
        report = report.merge(shap_df, on="feature", how="outer")

    return report.sort_values("gain_importance", ascending=False, na_position="last")
