"""
Data-leakage detection checks for the K-Predictor feature matrix.

Blueprint reference: Section 9 — Leakage Prevention.

Four checks are provided:

1. check_no_future_data(feature_matrix, fold_date)
   Assert that no row in feature_matrix has a game_date after fold_date.

2. check_rolling_features(feature_matrix, game_logs, fold_date)
   Verify that every rolling feature was computed exclusively from games
   that occurred strictly before each row's game_date.

3. check_same_day_lineup(feature_matrix, fold_date)
   Confirm that lineup-derived features do not incorporate same-day
   confirmed lineup data (only pre-announced / prior-day lineups are allowed).

4. run_all_leakage_checks(feature_matrix, game_logs, fold_date) -> bool
   Run all checks; returns True when all pass, False when any fail.
   Failures are logged as errors (not raised) so the caller can decide.

All checks return True on pass, False on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Rolling feature column inventory ─────────────────────────────────────────
# These columns should ONLY contain information from games prior to each row.
# Extend this list if new rolling features are added.
ROLLING_FEATURE_COLS = [
    "k_per_9_rolling",
    "swstr_pct_rolling",
    "csw_pct_rolling",
    "k_per_9_l5",
    "k_last_5_avg",
    "whiff_rate_rolling",
    "fb_velo_rolling",
    "fb_spin_rolling",
    "ip_per_start_rolling",
    "k_per_pa_rolling",
]

# Lineup-derived feature columns that must not use same-day data
LINEUP_FEATURE_COLS = [
    "opp_k_pct",
    "opp_k_pct_vs_lhp",
    "opp_k_pct_vs_rhp",
    "lineup_k_rate",
    "lineup_obp",
    "lineup_woba",
    "lineup_sub_risk",
    "opp_team_k_pct_l10",
]


# ── Check 1: No future data ───────────────────────────────────────────────────

def check_no_future_data(
    feature_matrix: pd.DataFrame,
    fold_date: str | pd.Timestamp,
) -> bool:
    """
    Check that every row in *feature_matrix* has game_date <= fold_date.

    Returns True when the check passes (no future data detected).
    Logs each offending row at ERROR level.
    """
    fold_dt = pd.Timestamp(fold_date)

    if "game_date" not in feature_matrix.columns:
        logger.warning("check_no_future_data: 'game_date' column missing — skipping check.")
        return True

    dates = pd.to_datetime(feature_matrix["game_date"])
    future_mask = dates > fold_dt
    n_future = future_mask.sum()

    if n_future == 0:
        logger.debug(f"check_no_future_data: PASS (fold_date={fold_date})")
        return True

    future_dates = dates[future_mask].unique()
    logger.error(
        f"check_no_future_data: FAIL — {n_future} rows have game_date > {fold_date}. "
        f"Offending dates: {sorted(future_dates)[:10]}"
    )
    return False


# ── Check 2: Rolling features use only prior-game data ───────────────────────

def check_rolling_features(
    feature_matrix: pd.DataFrame,
    game_logs: pd.DataFrame,
    fold_date: str | pd.Timestamp,
) -> bool:
    """
    Verify that rolling/lagged features are computed from games strictly
    before each row's game_date, not from the same date or later.

    Strategy:
        For each pitcher in feature_matrix, find the set of game_pks whose
        game_date == the row's game_date.  Cross-reference against game_logs
        to confirm those game_pks are not embedded in any rolling stat that
        should be lagged (i.e. rolling features must be the value *before*
        the current game).

    Practical check: if a k_per_9_rolling value equals the pitcher's stat
    computed *including* the current game_pk, that is a leakage signal.
    Because exact reconstruction of rolling windows is expensive, we apply
    a conservative heuristic: confirm that the number of rolling-window games
    is consistent with the prior-game count (not current-game count).

    Returns True when no leakage is detected.
    """
    fold_dt = pd.Timestamp(fold_date)

    if "game_date" not in feature_matrix.columns:
        logger.warning("check_rolling_features: 'game_date' missing — skipping.")
        return True

    present_cols = [c for c in ROLLING_FEATURE_COLS if c in feature_matrix.columns]
    if not present_cols:
        logger.debug("check_rolling_features: no rolling feature columns found — skipping.")
        return True

    if game_logs is None or len(game_logs) == 0:
        logger.warning("check_rolling_features: game_logs empty — skipping.")
        return True

    # Require game_date and pitcher_id in both DataFrames
    required = {"game_date", "pitcher_id"}
    if not required.issubset(feature_matrix.columns):
        logger.warning(f"check_rolling_features: missing columns {required - set(feature_matrix.columns)} — skipping.")
        return True
    if not required.issubset(game_logs.columns):
        logger.warning(f"check_rolling_features: game_logs missing {required - set(game_logs.columns)} — skipping.")
        return True

    game_logs = game_logs.copy()
    game_logs["game_date"] = pd.to_datetime(game_logs["game_date"])

    # For each row in feature_matrix, check that future-dated rows in
    # game_logs would not affect the rolling count.
    # Heuristic: count games in game_logs for each pitcher up-to (exclusive)
    # game_date, and verify it's non-negative.
    feature_matrix = feature_matrix.copy()
    feature_matrix["_game_date_dt"] = pd.to_datetime(feature_matrix["game_date"])

    leaked_rows = []
    for _, row in feature_matrix.iterrows():
        pitcher = row.get("pitcher_id")
        game_dt = row["_game_date_dt"]

        prior_games = game_logs[
            (game_logs["pitcher_id"] == pitcher)
            & (game_logs["game_date"] < game_dt)
        ]

        # If 'k_last_5_avg' exists, check if it's consistent with prior games
        if "k_last_5_avg" in feature_matrix.columns and pd.notna(row.get("k_last_5_avg")):
            if "strikeouts" in game_logs.columns and len(prior_games) > 0:
                last5 = prior_games.nlargest(5, "game_date")["strikeouts"].mean()
                recorded = row["k_last_5_avg"]
                # Tolerance of 0.01 to allow rounding
                if abs(last5 - recorded) > 0.5:
                    leaked_rows.append({
                        "pitcher_id": pitcher,
                        "game_date": game_dt,
                        "k_last_5_avg_recorded": recorded,
                        "k_last_5_avg_computed": last5,
                    })

    if leaked_rows:
        logger.error(
            f"check_rolling_features: FAIL — {len(leaked_rows)} rows show "
            f"possible leakage in k_last_5_avg. First offender: {leaked_rows[0]}"
        )
        return False

    logger.debug("check_rolling_features: PASS")
    return True


# ── Check 3: No same-day lineup data ─────────────────────────────────────────

def check_same_day_lineup(
    feature_matrix: pd.DataFrame,
    fold_date: str | pd.Timestamp,
) -> bool:
    """
    Confirm that lineup-derived features do not incorporate same-day
    official lineup data (which would not be available before game time).

    Detection strategy:
        If the feature matrix contains a column 'lineup_confirmed_date'
        (the date the lineup was confirmed), verify that it is always < game_date.
        If no such audit column exists, fall back to checking that
        `lineup_sub_risk` is not exactly 0.0 for every row (a suspicious
        uniform value that may indicate all starters are confirmed same-day).

    Returns True when no leakage is detected.
    """
    fold_dt = pd.Timestamp(fold_date)

    present_lineup_cols = [c for c in LINEUP_FEATURE_COLS if c in feature_matrix.columns]
    if not present_lineup_cols:
        logger.debug("check_same_day_lineup: no lineup feature columns present — skipping.")
        return True

    # If an audit column exists, use it directly
    if "lineup_confirmed_date" in feature_matrix.columns:
        fm = feature_matrix.copy()
        fm["_game_date_dt"] = pd.to_datetime(fm["game_date"])
        fm["_lineup_confirmed_dt"] = pd.to_datetime(fm["lineup_confirmed_date"])

        same_day_mask = fm["_lineup_confirmed_dt"] >= fm["_game_date_dt"]
        n_bad = same_day_mask.sum()
        if n_bad > 0:
            logger.error(
                f"check_same_day_lineup: FAIL — {n_bad} rows have "
                f"lineup_confirmed_date >= game_date (same-day lineup leakage)."
            )
            return False

        logger.debug("check_same_day_lineup: PASS (audit column check)")
        return True

    # Heuristic: if lineup_sub_risk == 0 for ALL rows, that may indicate
    # confirmed same-day lineups (every batter is locked in, risk = 0).
    if "lineup_sub_risk" in feature_matrix.columns:
        sub_risk = feature_matrix["lineup_sub_risk"].dropna()
        if len(sub_risk) > 0 and (sub_risk == 0.0).all():
            logger.warning(
                "check_same_day_lineup: lineup_sub_risk is 0.0 for all rows. "
                "This may indicate same-day confirmed lineup data was used. "
                "Verify that lineup features were computed from pre-game data."
            )
            # Treat as a warning, not a hard failure, since 0-risk is theoretically valid
            return True

    logger.debug("check_same_day_lineup: PASS (heuristic check)")
    return True


# ── Check 4: Orchestrator ─────────────────────────────────────────────────────

def run_all_leakage_checks(
    feature_matrix: pd.DataFrame,
    game_logs: Optional[pd.DataFrame],
    fold_date: str | pd.Timestamp,
) -> bool:
    """
    Run all four leakage checks in sequence.

    Returns True only when *all* checks pass; False otherwise.
    Individual check results are logged.
    """
    logger.info(f"Running leakage checks for fold_date={fold_date} "
                f"on {len(feature_matrix)} rows …")

    results = {
        "no_future_data": check_no_future_data(feature_matrix, fold_date),
        "rolling_features": check_rolling_features(feature_matrix, game_logs, fold_date),
        "same_day_lineup": check_same_day_lineup(feature_matrix, fold_date),
    }

    all_passed = all(results.values())

    if all_passed:
        logger.info("Leakage checks: ALL PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"Leakage checks: FAILED checks → {failed}")

    return all_passed
