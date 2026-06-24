"""
Blind test evaluation for the K-Predictor.

Samples 5 000 starts from 2024 post-All-Star-Break (ASB) games, stratified
by K/9 tier, with a fixed random seed. Evaluates the trained model against
actuals and against the three baselines defined in baselines.py.

All-Star Break cutoff: 2024-07-19 (first game after the break).

Usage:
    python -m src.evaluation.blind_test [--model-dir path] [--data-path path]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import GLOBAL_SEED, PREDICTIONS_DIR, PROJECT_ROOT
from src.evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_over_under_accuracy,
    compute_interval_calibration,
    compute_mae_by_tier,
    full_evaluation_report,
)
from src.evaluation.baselines import baseline_k9_x_ip, baseline_vegas_proxy, baseline_last5_avg

# ── Constants ─────────────────────────────────────────────────────────────────

# 2024 All-Star Break ends; first post-ASB game date
POST_ASB_CUTOFF = "2024-07-19"

# Target sample size for the blind test
BLIND_TEST_N = 5_000

# K/9 strata boundaries (right-exclusive on upper).  Used for stratification.
# Each row is assigned a stratum; sampling is proportional to stratum sizes
# but capped so total ~ BLIND_TEST_N.
K9_TIERS = {
    "low":    (0.0, 7.0),   # 0 – 6.99 K/9
    "medium": (7.0, 9.5),   # 7 – 9.49 K/9
    "high":   (9.5, 999.0), # 9.5+ K/9
}

# K/9 column preference (season > rolling > league avg)
_LEAGUE_K9 = 8.5


# ── Stratified sampling ───────────────────────────────────────────────────────

def _assign_k9_tier(df: pd.DataFrame) -> pd.Series:
    """Return a Series of tier labels for each row."""
    if "k_per_9_season" in df.columns:
        k9 = df["k_per_9_season"].fillna(_LEAGUE_K9)
    elif "k_per_9_rolling" in df.columns:
        k9 = df["k_per_9_rolling"].fillna(_LEAGUE_K9)
    else:
        k9 = pd.Series(_LEAGUE_K9, index=df.index)

    tiers = pd.Series("medium", index=df.index, dtype=object)
    for name, (lo, hi) in K9_TIERS.items():
        tiers[k9.between(lo, hi, inclusive="left")] = name
    return tiers


def stratified_sample(df: pd.DataFrame, n: int = BLIND_TEST_N,
                       seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """
    Draw *n* rows stratified by K/9 tier (proportional allocation).
    Uses fixed *seed* for reproducibility.
    """
    df = df.copy()
    df["_tier"] = _assign_k9_tier(df)

    tier_counts = df["_tier"].value_counts()
    total = len(df)

    rng = np.random.RandomState(seed)
    frames = []
    remaining = n

    tiers = list(tier_counts.index)
    for idx, tier in enumerate(tiers):
        tier_df = df[df["_tier"] == tier]
        if idx == len(tiers) - 1:
            # Last tier gets whatever is left to ensure exact total
            k = min(remaining, len(tier_df))
        else:
            prop = tier_counts[tier] / total
            k = min(int(round(n * prop)), len(tier_df), remaining)

        if k > 0:
            sample = tier_df.sample(n=k, random_state=rng, replace=False)
            frames.append(sample)
            remaining -= k

        if remaining <= 0:
            break

    result = pd.concat(frames).drop(columns=["_tier"]).reset_index(drop=True)
    logger.info(
        f"Blind-test sample: {len(result)} rows "
        f"(tiers: {df['_tier'].value_counts().to_dict()})"
    )
    return result


# ── Baseline evaluation helper ────────────────────────────────────────────────

def _evaluate_one(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                  lower: np.ndarray = None, upper: np.ndarray = None) -> dict:
    """Return a standard metrics dict for one model / baseline."""
    report = full_evaluation_report(y_true, y_pred, lower, upper)
    report["model"] = name
    return report


# ── Main blind test ───────────────────────────────────────────────────────────

def run_blind_test(data_path: Path = None, model_dir: Path = None) -> dict:
    """
    Execute the blind test:
    1. Load 2024 post-ASB data.
    2. Stratified-sample 5 000 starts (fixed seed).
    3. Generate predictions from the trained model.
    4. Evaluate model vs. B1, B2, B3 baselines.
    5. Return and persist results.
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    from src.model.train import prepare_training_data, load_feature_columns

    logger.info("Loading feature data for blind test …")
    df, feature_cols = prepare_training_data(data_path)

    # Filter to 2024 post-ASB
    df["game_date"] = pd.to_datetime(df["game_date"])
    post_asb = df[df["game_date"] >= POST_ASB_CUTOFF].copy()

    if len(post_asb) == 0:
        logger.warning(
            "No 2024 post-ASB rows found in feature data. "
            "Running blind test on the full dataset instead."
        )
        post_asb = df.copy()

    logger.info(f"Post-ASB pool: {len(post_asb)} rows")

    # ── Stratified sample ─────────────────────────────────────────────────────
    n = min(BLIND_TEST_N, len(post_asb))
    sample = stratified_sample(post_asb, n=n, seed=GLOBAL_SEED)

    y_true = sample["actual_strikeouts"].values

    # ── Model predictions ─────────────────────────────────────────────────────
    from src.model.predict import predict, find_latest_model

    metadata = find_latest_model(model_dir)
    pred_df = predict(sample, metadata=metadata)
    y_model = pred_df["predicted_strikeouts"].values
    y_lower = pred_df["pred_ci_lower"].values
    y_upper = pred_df["pred_ci_upper"].values

    # ── Baseline predictions ──────────────────────────────────────────────────
    y_b1 = baseline_k9_x_ip(sample)
    y_b2 = baseline_vegas_proxy(sample)
    y_b3 = baseline_last5_avg(sample)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = {
        "model": _evaluate_one("lgbm_stacked", y_true, y_model, y_lower, y_upper),
        "baseline_b1_k9_x_ip": _evaluate_one("B1: K/9 × IP", y_true, y_b1),
        "baseline_b2_vegas_proxy": _evaluate_one("B2: Vegas proxy", y_true, y_b2),
        "baseline_b3_last5_avg": _evaluate_one("B3: Last-5 avg", y_true, y_b3),
    }

    # Print summary table
    header = f"{'Model':<25} {'MAE':>7} {'RMSE':>7} {'O/U@6.5':>9} {'O/U@7.5':>9} {'Calib90':>9}"
    logger.info("=== Blind Test Results (2024 post-ASB, n={}) ===".format(len(sample)))
    logger.info(header)
    logger.info("-" * len(header))
    for key, r in results.items():
        ou65 = r["over_under_accuracy"].get("6.5", float("nan"))
        ou75 = r["over_under_accuracy"].get("7.5", float("nan"))
        cal = r.get("interval_calibration_90", float("nan"))
        logger.info(
            f"{r['model']:<25} {r['mae']:>7.3f} {r['rmse']:>7.3f}"
            f" {ou65:>9.3f} {ou75:>9.3f} {cal:>9.3f}"
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    out_dir = PREDICTIONS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "blind_test_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Blind test report saved → {report_path}")

    # Save the sampled rows with predictions for further analysis
    sample_out = sample[["pitcher_id", "game_date", "game_pk", "actual_strikeouts"]].copy()
    sample_out["pred_model"] = y_model
    sample_out["pred_lower"] = y_lower
    sample_out["pred_upper"] = y_upper
    sample_out["pred_b1"] = y_b1
    sample_out["pred_b2"] = y_b2
    sample_out["pred_b3"] = y_b3
    sample_out.to_parquet(out_dir / "blind_test_predictions.parquet", index=False)
    logger.info(f"Blind test predictions saved → {out_dir / 'blind_test_predictions.parquet'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run K-Predictor blind test (2024 post-ASB)")
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Path to combined feature parquet (optional)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Directory containing model metadata (optional)")
    args = parser.parse_args()

    run_blind_test(data_path=args.data_path, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
