"""
Missing data imputation following the blueprint's hierarchy:
1. Expand lookback window
2. Use prior-season data
3. Use league average
4. Let LightGBM handle natively (NaN)
"""
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUE_AVG

# Features where NaN should be passed to LightGBM (don't impute)
NATIVE_NAN_FEATURES = {
    "ump_k_boost", "catcher_framing_runs",
    "battery_k_rate_delta", "catcher_game_calling_aggression",
}

# Feature -> league average key mapping
FEATURE_LEAGUE_AVG = {
    "k_per_9_rolling": "k_per_9",
    "swstr_pct": "swstr_pct",
    "csw_pct": "csw_pct",
    "fb_spin_rate": "fb_spin_rate",
    "ip_per_start": "ip_per_start",
    "opp_team_k_rate_vs_hand": "team_k_pct",
    "opp_o_swing_pct": "o_swing_pct",
    "opp_z_contact_pct": "z_contact_pct",
    "opp_contact_rate": "contact_pct",
    "two_strike_putaway_rate": "putaway_rate",
    "projected_lineup_weighted_k_rate": "team_k_pct",
    "opp_whiff_vs_pitch_types": "swstr_pct",
    "opp_chase_vs_velo_band": "o_swing_pct",
    "lineup_sub_risk": "lineup_sub_risk",
    "park_k_factor": "park_k_factor",
    "temperature_f": "dome_temp_f",
    "humidity_pct": "dome_humidity_pct",
}

# Hard-coded fallbacks for features without a league-avg key
FEATURE_HARD_DEFAULTS = {
    "k_rate_last_5": None,          # Defer to k_per_9_rolling value
    "recent_pitch_count_trend": 0.0,
    "days_rest": 5.0,
    "fb_velo_avg": 93.0,
    "fb_velo_max": 96.0,
    "vaa_fastball": -5.0,
    "ivb_delta": 0.0,
    "hb_delta": 0.0,
    "release_point_consistency": 0.15,
    "arm_slot_drift": 0.0,
    "tunneling_efficiency": 0.5,
    "pitch_mix_k_profile": 0.25,
    "lineup_handedness_stack": 4,
    "opp_plate_discipline_variance": 0.05,
    "matchup_familiarity": 0,
    "opp_travel_fatigue": 0,
    "opp_game_importance": 1,
    "pitcher_leash": 1.0,
    "battery_k_rate_together": None,  # Defer to k_per_9_rolling
}


def impute_features(row: dict) -> dict:
    """
    Apply imputation rules to a feature dict.

    Priority:
      1. NATIVE_NAN_FEATURES: leave as NaN for LightGBM
      2. FEATURE_LEAGUE_AVG: fill with LEAGUE_AVG[key]
      3. FEATURE_HARD_DEFAULTS: fill with a sensible constant
      4. Otherwise: leave as NaN (LightGBM handles natively)
    """
    imputed = {}
    for feat, val in row.items():
        if feat in NATIVE_NAN_FEATURES:
            # Always keep NaN — let LightGBM handle natively
            imputed[feat] = val
            continue

        if _is_missing(val):
            # Step 1: try league average mapping
            league_key = FEATURE_LEAGUE_AVG.get(feat)
            if league_key and league_key in LEAGUE_AVG:
                imputed[feat] = LEAGUE_AVG[league_key]
                continue

            # Step 2: try hard default
            hard_default = FEATURE_HARD_DEFAULTS.get(feat)
            if hard_default is not None:
                imputed[feat] = hard_default
                continue

            # Step 3: deferred cross-feature reference
            if feat == "k_rate_last_5" and not _is_missing(row.get("k_per_9_rolling")):
                imputed[feat] = row["k_per_9_rolling"]
                continue

            if feat == "battery_k_rate_together" and not _is_missing(row.get("k_per_9_rolling")):
                imputed[feat] = row["k_per_9_rolling"]
                continue

            # Step 4: keep NaN — LightGBM will handle
            imputed[feat] = val
        else:
            imputed[feat] = val

    return imputed


def _is_missing(val) -> bool:
    """Check whether a value is missing (None, NaN, or empty string)."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply imputation row-by-row to a feature DataFrame.
    Returns a new DataFrame with missing values filled per blueprint rules.
    """
    if df.empty:
        return df

    records = df.to_dict(orient="records")
    imputed_records = [impute_features(row) for row in records]
    return pd.DataFrame(imputed_records, index=df.index)


def get_feature_fill_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic: return a report showing NaN counts and fill rates per column.
    Useful for monitoring data quality in the feature pipeline.
    """
    total = len(df)
    report_rows = []
    for col in df.columns:
        nan_count = df[col].isna().sum()
        fill_pct = 100.0 * (1 - nan_count / total) if total > 0 else 0.0
        is_native_nan = col in NATIVE_NAN_FEATURES
        report_rows.append({
            "feature": col,
            "nan_count": nan_count,
            "fill_pct": round(fill_pct, 1),
            "native_nan": is_native_nan,
        })
    return pd.DataFrame(report_rows).sort_values("fill_pct")
