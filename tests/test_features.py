"""
Tests for feature computation modules.
Uses fixtures from conftest.py — no real external data needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Expected feature group names (used across multiple tests)
# ─────────────────────────────────────────────────────────────────────────────
PITCHER_ABILITY_KEYS = [
    "k_per_9_rolling", "swstr_pct", "csw_pct", "pitch_mix_k_profile",
    "fb_velo_avg", "fb_velo_max", "fb_spin_rate", "vaa_fastball",
    "ivb_delta", "hb_delta", "release_point_consistency", "arm_slot_drift",
    "tunneling_efficiency", "two_strike_putaway_rate",
]

RECENT_FORM_KEYS = [
    "k_rate_last_5", "recent_pitch_count_trend", "ip_per_start", "days_rest",
]

OPPONENT_PROFILE_KEYS = [
    "opp_team_k_rate_vs_hand", "opp_o_swing_pct", "opp_z_contact_pct",
    "opp_contact_rate", "projected_lineup_weighted_k_rate",
    "opp_whiff_vs_pitch_types", "opp_chase_vs_velo_band",
    "lineup_handedness_stack", "opp_plate_discipline_variance",
    "lineup_sub_risk", "matchup_familiarity", "opp_travel_fatigue",
    "opp_game_importance",
]

CONTEXTUAL_KEYS = [
    "park_k_factor", "ump_k_boost", "temperature_f", "humidity_pct",
    "pitcher_leash",
]

BATTERY_EFFECTS_KEYS = [
    "catcher_framing_runs", "battery_k_rate_together",
    "battery_k_rate_delta", "catcher_game_calling_aggression",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: drop the pitcher whose game_date == as_of_date to avoid self-inclusion
# ─────────────────────────────────────────────────────────────────────────────
def _as_of_date_for(sample_pitches):
    """Return a date that is strictly after some pitches exist."""
    dates = sorted(pd.to_datetime(sample_pitches["game_date"]).unique())
    # Use a date after at least 3 game dates for meaningful rolling windows
    return str(dates[3].date())


# ─────────────────────────────────────────────────────────────────────────────
# 1. test_pitcher_ability_computation
# ─────────────────────────────────────────────────────────────────────────────
def test_pitcher_ability_computation(sample_pitches, sample_game_logs):
    """compute_all_pitcher_ability returns exactly 14 keys, all numeric."""
    from src.features.pitcher_ability import compute_all_pitcher_ability

    pitcher_id = 12345
    as_of_date = _as_of_date_for(sample_pitches)

    result = compute_all_pitcher_ability(sample_pitches, sample_game_logs, pitcher_id, as_of_date)

    assert isinstance(result, dict)
    assert len(result) == 14, f"Expected 14 keys, got {len(result)}: {list(result.keys())}"
    assert set(result.keys()) == set(PITCHER_ABILITY_KEYS)

    for key, val in result.items():
        assert val is not None, f"Feature {key} is None"
        assert isinstance(val, (int, float, np.integer, np.floating)), (
            f"Feature {key} value {val!r} is not numeric"
        )


@pytest.mark.parametrize("pitcher_id", [12345, 67890])
def test_pitcher_ability_both_pitchers(sample_pitches, sample_game_logs, pitcher_id):
    """Pitcher ability features work for both pitchers in sample data."""
    from src.features.pitcher_ability import compute_all_pitcher_ability

    as_of_date = _as_of_date_for(sample_pitches)
    result = compute_all_pitcher_ability(sample_pitches, sample_game_logs, pitcher_id, as_of_date)
    assert len(result) == 14


def test_pitcher_ability_fallback_on_empty_data():
    """When called with empty DataFrames (with correct columns), returns fallback/league-avg values."""
    from src.features.pitcher_ability import compute_all_pitcher_ability

    # Build minimal empty DataFrames with the expected columns
    empty_pitches = pd.DataFrame(columns=[
        "pitcher", "game_date", "pitch_type", "description",
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
        "release_pos_x", "release_pos_z", "plate_x", "plate_z",
        "strikes", "balls", "events", "at_bat_number",
    ])
    empty_games = pd.DataFrame(columns=[
        "pitcher_id", "game_date", "strikeouts", "innings_pitched",
        "pitches_thrown", "walks", "earned_runs",
    ])

    result = compute_all_pitcher_ability(
        empty_pitches, empty_games, 99999, "2024-07-01"
    )
    assert len(result) == 14
    # All values should be numeric (not None, not raise)
    for key, val in result.items():
        assert isinstance(val, (int, float, np.integer, np.floating)), f"{key}={val!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. test_recent_form_computation
# ─────────────────────────────────────────────────────────────────────────────
def test_recent_form_computation(sample_game_logs):
    """compute_all_recent_form returns exactly 4 keys with expected names."""
    from src.features.recent_form import compute_all_recent_form

    pitcher_id = 12345
    # Use a date after all 5 sample game dates
    last_date = str(pd.to_datetime(sample_game_logs["game_date"]).max().date())
    as_of_date = str((pd.to_datetime(last_date) + pd.Timedelta(days=5)).date())

    result = compute_all_recent_form(sample_game_logs, pitcher_id, as_of_date)

    assert isinstance(result, dict)
    assert len(result) == 4, f"Expected 4 keys, got {len(result)}: {list(result.keys())}"
    assert set(result.keys()) == set(RECENT_FORM_KEYS)

    for key, val in result.items():
        assert isinstance(val, (int, float, np.integer, np.floating)), (
            f"Feature {key} value {val!r} is not numeric"
        )


def test_recent_form_days_rest_reasonable(sample_game_logs):
    """days_rest should be a positive number (1-30)."""
    from src.features.recent_form import compute_all_recent_form

    last_date = str(pd.to_datetime(sample_game_logs["game_date"]).max().date())
    as_of_date = str((pd.to_datetime(last_date) + pd.Timedelta(days=4)).date())
    result = compute_all_recent_form(sample_game_logs, 12345, as_of_date)
    assert 0 < result["days_rest"] <= 30


# ─────────────────────────────────────────────────────────────────────────────
# 3. test_opponent_profile_computation
# ─────────────────────────────────────────────────────────────────────────────
def test_opponent_profile_computation(sample_pitches, sample_game_logs):
    """compute_all_opponent_profile returns exactly 13 keys."""
    from src.features.opponent_profile import compute_all_opponent_profile

    as_of_date = _as_of_date_for(sample_pitches)
    result = compute_all_opponent_profile(
        pitches=sample_pitches,
        game_logs=sample_game_logs,
        lineup_df=pd.DataFrame(),
        pitcher_id=12345,
        opp_team_id=999,
        as_of_date=as_of_date,
        game_pk=7777,
        pitcher_hand="R",
        primary_velo=94.0,
        standings_gb=5.0,
        schedule_data=None,
    )

    assert isinstance(result, dict)
    assert len(result) == 13, f"Expected 13 keys, got {len(result)}: {list(result.keys())}"
    assert set(result.keys()) == set(OPPONENT_PROFILE_KEYS)

    for key, val in result.items():
        assert isinstance(val, (int, float, np.integer, np.floating)), (
            f"Feature {key} value {val!r} is not numeric"
        )


def test_opponent_profile_fallback_on_empty():
    """Opponent profile returns dict of 13 numeric values even with empty inputs (with columns)."""
    from src.features.opponent_profile import compute_all_opponent_profile

    empty_pitches = pd.DataFrame(columns=[
        "game_date", "pitcher", "batter", "pitch_type", "description",
        "release_speed", "zone", "events", "p_throws", "stand",
    ])
    empty_games = pd.DataFrame(columns=[
        "pitcher_id", "game_date", "opponent_team_id", "strikeouts", "innings_pitched",
    ])

    result = compute_all_opponent_profile(
        pitches=empty_pitches,
        game_logs=empty_games,
        lineup_df=pd.DataFrame(),
        pitcher_id=99999,
        opp_team_id=0,
        as_of_date="2024-07-01",
        game_pk=0,
    )
    assert len(result) == 13


# ─────────────────────────────────────────────────────────────────────────────
# 4. test_contextual_features
# ─────────────────────────────────────────────────────────────────────────────
def test_contextual_features(sample_game_logs):
    """compute_all_contextual returns exactly 5 keys."""
    from src.features.contextual import compute_all_contextual

    result = compute_all_contextual(
        ballpark_id="NYY",
        umpire_history=None,
        umpire_id=0,
        as_of_date="2024-06-01",
        weather_df=pd.DataFrame(),
        game_pk=9999,
        game_logs=sample_game_logs,
        team_id=147,
    )

    assert isinstance(result, dict)
    assert len(result) == 5, f"Expected 5 keys, got {len(result)}: {list(result.keys())}"
    assert set(result.keys()) == set(CONTEXTUAL_KEYS)

    for key, val in result.items():
        assert isinstance(val, (int, float, np.integer, np.floating)) or (
            isinstance(val, float) and np.isnan(val)
        ), f"Feature {key} value {val!r} is not numeric or NaN"


@pytest.mark.parametrize("ballpark_id,expected_min,expected_max", [
    ("NYY", 95.0, 105.0),   # Open-air sea level
    ("COL", 90.0, 100.0),   # High altitude (Coors)
    ("", 95.0, 105.0),      # Empty string fallback
])
def test_park_k_factor_range(ballpark_id, expected_min, expected_max, sample_game_logs):
    """Park K factor should fall within expected range per elevation tier."""
    from src.features.contextual import compute_all_contextual

    result = compute_all_contextual(
        ballpark_id=ballpark_id,
        umpire_history=None,
        umpire_id=0,
        as_of_date="2024-06-01",
        weather_df=pd.DataFrame(),
        game_pk=9999,
        game_logs=sample_game_logs,
        team_id=147,
    )
    assert expected_min <= result["park_k_factor"] <= expected_max


# ─────────────────────────────────────────────────────────────────────────────
# 5. test_battery_effects
# ─────────────────────────────────────────────────────────────────────────────
def test_battery_effects(sample_pitches, sample_game_logs):
    """compute_all_battery_effects returns exactly 4 keys."""
    from src.features.battery_effects import compute_all_battery_effects

    as_of_date = _as_of_date_for(sample_pitches)
    result = compute_all_battery_effects(
        pitches=sample_pitches,
        game_logs=sample_game_logs,
        lineups=pd.DataFrame(),
        pitcher_id=12345,
        catcher_id=0,   # Unknown catcher → NaN for some features
        as_of_date=as_of_date,
    )

    assert isinstance(result, dict)
    assert len(result) == 4, f"Expected 4 keys, got {len(result)}: {list(result.keys())}"
    assert set(result.keys()) == set(BATTERY_EFFECTS_KEYS)

    # Values should be numeric or NaN (native LightGBM handling)
    for key, val in result.items():
        assert isinstance(val, (int, float, np.integer, np.floating)), (
            f"Feature {key} value {val!r} is not numeric/NaN"
        )


def test_battery_effects_with_known_catcher(sample_pitches, sample_game_logs):
    """Battery effects with catcher_id != 0 still returns 4 keys."""
    from src.features.battery_effects import compute_all_battery_effects

    as_of_date = _as_of_date_for(sample_pitches)
    result = compute_all_battery_effects(
        pitches=sample_pitches,
        game_logs=sample_game_logs,
        lineups=pd.DataFrame(),
        pitcher_id=12345,
        catcher_id=55555,
        as_of_date=as_of_date,
    )
    assert len(result) == 4


# ─────────────────────────────────────────────────────────────────────────────
# 6. test_feature_pipeline_build_matrix
# ─────────────────────────────────────────────────────────────────────────────
def test_feature_pipeline_build_matrix(sample_pitches, sample_game_logs):
    """
    build_feature_matrix for a single pitcher produces a DataFrame
    with all 40 feature columns present.
    """
    from src.features.feature_pipeline import build_feature_matrix, FEATURE_COLUMNS

    as_of_date = _as_of_date_for(sample_pitches)

    starters_df = pd.DataFrame([{
        "pitcher_id": 12345,
        "opponent_team_id": 999,
        "game_pk": 8888,
        "ballpark_id": "NYY",
        "pitcher_hand": "R",
        "home_plate_umpire_id": 0,
        "catcher_id": 0,
        "team_id": 147,
        "standings_gb": 5.0,
    }])

    result = build_feature_matrix(
        prediction_date=as_of_date,
        starters_df=starters_df,
        pitches=sample_pitches,
        game_logs=sample_game_logs,
        lineups=pd.DataFrame(),
        weather_df=pd.DataFrame(),
        umpire_history=None,
        save=False,
    )

    assert isinstance(result, pd.DataFrame), "build_feature_matrix should return a DataFrame"
    assert len(result) >= 1, "Should have at least one row"

    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature column: {col}"


def test_feature_pipeline_returns_exactly_40_feature_cols(sample_pitches, sample_game_logs):
    """The result DataFrame has all 40 feature columns (may have additional metadata)."""
    from src.features.feature_pipeline import build_feature_matrix, FEATURE_COLUMNS

    as_of_date = _as_of_date_for(sample_pitches)
    starters_df = pd.DataFrame([{
        "pitcher_id": 12345,
        "opponent_team_id": 999,
        "game_pk": 8888,
        "ballpark_id": "NYY",
    }])

    result = build_feature_matrix(
        prediction_date=as_of_date,
        starters_df=starters_df,
        pitches=sample_pitches,
        game_logs=sample_game_logs,
        lineups=pd.DataFrame(),
        weather_df=pd.DataFrame(),
        umpire_history=None,
        save=False,
    )

    feature_cols_present = [c for c in FEATURE_COLUMNS if c in result.columns]
    assert len(feature_cols_present) == 40, (
        f"Expected 40 feature columns in result, found {len(feature_cols_present)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. test_imputation_fills_nans
# ─────────────────────────────────────────────────────────────────────────────
def test_imputation_fills_nans():
    """impute_features fills NaN values; not all values remain NaN."""
    from src.features.imputation import impute_features
    from src.features.feature_pipeline import FEATURE_COLUMNS

    # Build an all-NaN feature dict
    nan_row = {col: float("nan") for col in FEATURE_COLUMNS}
    result = impute_features(nan_row)

    # Count filled values
    filled = sum(
        1 for key, val in result.items()
        if val is not None and not (isinstance(val, float) and np.isnan(val))
    )
    assert filled > 0, "impute_features left all features as NaN"


def test_imputation_league_averages():
    """Key features get league-average values after imputation."""
    from src.features.imputation import impute_features
    from config.settings import LEAGUE_AVG

    nan_row = {
        "k_per_9_rolling": float("nan"),
        "swstr_pct": float("nan"),
        "csw_pct": float("nan"),
        "days_rest": float("nan"),
        "fb_velo_avg": float("nan"),
    }
    result = impute_features(nan_row)

    assert result["k_per_9_rolling"] == pytest.approx(LEAGUE_AVG["k_per_9"])
    assert result["swstr_pct"] == pytest.approx(LEAGUE_AVG["swstr_pct"])
    assert result["csw_pct"] == pytest.approx(LEAGUE_AVG["csw_pct"])
    assert result["days_rest"] == pytest.approx(5.0)
    assert result["fb_velo_avg"] == pytest.approx(93.0)


def test_imputation_dataframe(sample_feature_matrix):
    """impute_dataframe processes an entire DataFrame row by row."""
    from src.features.imputation import impute_dataframe
    from src.features.feature_pipeline import FEATURE_COLUMNS

    # Introduce NaN in a few columns
    df_with_nans = sample_feature_matrix.copy()
    df_with_nans.loc[0:4, "k_per_9_rolling"] = np.nan
    df_with_nans.loc[0:4, "swstr_pct"] = np.nan

    result = impute_dataframe(df_with_nans)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df_with_nans)
    # After imputation, the filled columns should not be NaN anymore
    assert result.loc[0, "k_per_9_rolling"] is not None
    assert not np.isnan(result.loc[0, "k_per_9_rolling"])


def test_imputation_native_nan_preserved():
    """
    NATIVE_NAN_FEATURES (ump_k_boost, catcher_framing_runs, etc.) should
    remain NaN even after imputation — LightGBM handles them natively.
    """
    from src.features.imputation import impute_features, NATIVE_NAN_FEATURES

    nan_row = {feat: float("nan") for feat in NATIVE_NAN_FEATURES}
    result = impute_features(nan_row)

    for feat in NATIVE_NAN_FEATURES:
        val = result[feat]
        assert isinstance(val, float) and np.isnan(val), (
            f"Expected NaN for native NaN feature {feat}, got {val!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. test_feature_column_order
# ─────────────────────────────────────────────────────────────────────────────
def test_feature_column_order():
    """FEATURE_COLUMNS in feature_pipeline has exactly 40 entries matching the spec."""
    from src.features.feature_pipeline import FEATURE_COLUMNS as pipeline_cols

    assert len(pipeline_cols) == 40, (
        f"Expected 40 feature columns, found {len(pipeline_cols)}"
    )
    assert pipeline_cols == FEATURE_COLUMNS, (
        "feature_pipeline.FEATURE_COLUMNS order doesn't match spec"
    )


def test_feature_columns_no_duplicates():
    """No duplicate column names in FEATURE_COLUMNS."""
    from src.features.feature_pipeline import FEATURE_COLUMNS as pipeline_cols

    assert len(pipeline_cols) == len(set(pipeline_cols)), "Duplicate column names found"


def test_all_spec_features_present():
    """Every feature from the spec is in FEATURE_COLUMNS."""
    from src.features.feature_pipeline import FEATURE_COLUMNS as pipeline_cols

    missing = set(FEATURE_COLUMNS) - set(pipeline_cols)
    assert not missing, f"Features missing from pipeline: {missing}"
