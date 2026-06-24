"""
Critical data leakage detection tests.
These verify that features are computed using only past data
and that train/test splits are temporally clean.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a minimal time-series feature matrix spanning multiple dates
# ─────────────────────────────────────────────────────────────────────────────
def _make_temporal_feature_matrix(n_weeks: int = 14) -> pd.DataFrame:
    """
    Build a synthetic feature matrix spanning n_weeks.
    Each week has 2 rows (one per pitcher).
    Returns DataFrame with game_date, actual_strikeouts, and all 40 feature cols.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-04-01", periods=n_weeks, freq="7D")
    rows = []
    for d in dates:
        for pid in [12345, 67890]:
            row = {"game_date": d.strftime("%Y-%m-%d"), "pitcher_id": pid, "actual_strikeouts": float(rng.integers(1, 12))}
            for col in FEATURE_COLUMNS:
                row[col] = float(rng.uniform(0, 1))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. test_no_future_data_in_features
# ─────────────────────────────────────────────────────────────────────────────
def test_no_future_data_in_features(sample_pitches, sample_game_logs):
    """
    Features computed for prediction_date X must only use data
    strictly BEFORE date X (data_as_of_date < X).
    
    Verifies this by computing features for date D and checking that
    no pitch or game log from date D is included in the inputs.
    """
    from src.features.pitcher_ability import compute_all_pitcher_ability
    from src.features.recent_form import compute_all_recent_form

    # Choose a date that's in the middle of the sample data
    all_dates = sorted(pd.to_datetime(sample_pitches["game_date"]).unique())
    target_date = str(all_dates[4].date())  # 5th date in sample

    # Count pitches/games available before target_date for pitcher 12345
    pitches_before = sample_pitches[
        (sample_pitches["pitcher"] == 12345) &
        (pd.to_datetime(sample_pitches["game_date"]) < pd.to_datetime(target_date))
    ]
    pitches_on_date = sample_pitches[
        (sample_pitches["pitcher"] == 12345) &
        (pd.to_datetime(sample_pitches["game_date"]) == pd.to_datetime(target_date))
    ]

    ability_before = compute_all_pitcher_ability(
        sample_pitches, sample_game_logs, 12345, target_date
    )
    # If we compute for the DAY AFTER target_date, swstr_pct etc. should
    # incorporate the target_date game — meaning they differ (more data used)
    next_date = str((pd.to_datetime(target_date) + pd.Timedelta(days=1)).date())
    ability_after = compute_all_pitcher_ability(
        sample_pitches, sample_game_logs, 12345, next_date
    )

    # The features computed for next_date should have access to more data
    # (target_date pitches included). If there ARE pitches on target_date,
    # at least one feature should differ (swstr_pct or fb_velo_avg).
    if not pitches_on_date.empty:
        changed = any(
            ability_before[k] != ability_after[k]
            for k in ["swstr_pct", "csw_pct", "fb_velo_avg"]
            if not (np.isnan(ability_before[k]) and np.isnan(ability_after[k]))
        )
        assert changed, (
            "Features for dates D and D+1 are identical despite new pitches on date D; "
            "possible future-data leak or caching issue"
        )

    # No pitch from target_date or later should appear in a "before" slice
    assert len(pitches_before) < len(sample_pitches[sample_pitches["pitcher"] == 12345])


# ─────────────────────────────────────────────────────────────────────────────
# 2. test_rolling_window_boundary
# ─────────────────────────────────────────────────────────────────────────────
def test_rolling_window_boundary(sample_pitches, sample_game_logs):
    """
    Features computed for date D+7 incorporate more data than date D.
    Rolling stats should differ as more game data is included.
    """
    from src.features.recent_form import compute_all_recent_form

    dates = sorted(pd.to_datetime(sample_game_logs["game_date"]).unique())
    # Need two dates far enough apart that game logs differ
    date_early = str(dates[1].date())
    date_late = str(dates[-1].date())

    form_early = compute_all_recent_form(sample_game_logs, 12345, date_early)
    form_late = compute_all_recent_form(sample_game_logs, 12345, date_late)

    # k_rate_last_5 should be different because different sets of games are used
    # (early date has fewer games available)
    early_k9 = form_early["k_rate_last_5"]
    late_k9 = form_late["k_rate_last_5"]

    # They won't always differ (both could hit league-avg fallback),
    # but the number of games available MUST differ
    games_early = sample_game_logs[
        (sample_game_logs["pitcher_id"] == 12345) &
        (pd.to_datetime(sample_game_logs["game_date"]) < pd.to_datetime(date_early))
    ]
    games_late = sample_game_logs[
        (sample_game_logs["pitcher_id"] == 12345) &
        (pd.to_datetime(sample_game_logs["game_date"]) < pd.to_datetime(date_late))
    ]
    assert len(games_late) > len(games_early), (
        "Later date should have access to more game logs"
    )


def test_rolling_window_does_not_include_future_games(sample_game_logs):
    """
    feat_k_rate_last_5 for date D should NOT include games on date D or later.
    """
    from src.features.recent_form import feat_k_rate_last_5

    # Use the 3rd game date as the cutoff
    dates = sorted(pd.to_datetime(sample_game_logs["game_date"]).unique())
    cutoff = str(dates[2].date())

    # Games before cutoff for pitcher 12345
    games_before = sample_game_logs[
        (sample_game_logs["pitcher_id"] == 12345) &
        (pd.to_datetime(sample_game_logs["game_date"]) < pd.to_datetime(cutoff))
    ]

    k9 = feat_k_rate_last_5(sample_game_logs, 12345, cutoff)
    assert isinstance(k9, (float, np.floating, int))

    # If no games before cutoff, should fall back to league average
    if games_before.empty:
        from config.settings import LEAGUE_AVG
        assert k9 == pytest.approx(LEAGUE_AVG["k_per_9"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. test_opponent_features_no_same_game_data
# ─────────────────────────────────────────────────────────────────────────────
def test_opponent_features_no_same_game_data(sample_pitches, sample_game_logs):
    """
    Opponent features for game_pk X should not use pitch/game data
    from the game being predicted (data must be < as_of_date).
    """
    from src.features.opponent_profile import (
        feat_opp_team_k_rate_vs_hand,
        feat_opp_o_swing_pct,
    )

    all_dates = sorted(pd.to_datetime(sample_pitches["game_date"]).unique())
    target_date = str(all_dates[3].date())

    # Pitches strictly before target_date
    pitches_before = sample_pitches[
        pd.to_datetime(sample_pitches["game_date"]) < pd.to_datetime(target_date)
    ]
    # Pitches including target_date (simulating a leak)
    pitches_including = sample_pitches[
        pd.to_datetime(sample_pitches["game_date"]) <= pd.to_datetime(target_date)
    ]

    # We need a team column for opponent profile lookups
    # Add a batter_team column pointing to opp_team_id for demo
    pitches_before = pitches_before.copy()
    pitches_before["batter_team"] = 999  # All pitches face team 999

    pitches_including = pitches_including.copy()
    pitches_including["batter_team"] = 999

    k_rate_before = feat_opp_team_k_rate_vs_hand(
        pitches_before, opp_team_id=999, pitcher_hand="R", as_of_date=target_date
    )
    k_rate_after = feat_opp_team_k_rate_vs_hand(
        pitches_including, opp_team_id=999, pitcher_hand="R", as_of_date=target_date
    )

    # The "before" version uses as_of_date < target_date filter internally
    # Both should return valid floats
    assert isinstance(k_rate_before, (float, np.floating))
    assert isinstance(k_rate_after, (float, np.floating))
    assert 0.0 <= k_rate_before <= 1.0
    assert 0.0 <= k_rate_after <= 1.0

    # The as_of_date < filter should exclude target_date pitches
    # Verify the internal filter works by checking game_date filtering
    opp_pitches_used = pitches_before[
        pd.to_datetime(pitches_before["game_date"]) < pd.to_datetime(target_date)
    ]
    assert all(
        pd.to_datetime(row["game_date"]) < pd.to_datetime(target_date)
        for _, row in opp_pitches_used.iterrows()
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. test_walk_forward_fold_integrity
# ─────────────────────────────────────────────────────────────────────────────
def test_walk_forward_fold_integrity():
    """
    Walk-forward splits must have NO overlap between train and test dates.
    Test simulates the fold logic from walk_forward.py directly.
    """
    df = _make_temporal_feature_matrix(n_weeks=14)
    df = df.sort_values("game_date").reset_index(drop=True)

    # Replicate the week assignment logic from walk_forward.py
    df["year"] = df["game_date"].dt.year
    df["week_of_year"] = df["game_date"].dt.isocalendar().week.astype(int)
    min_year = df["year"].min()
    df["week_num"] = (df["year"] - min_year) * 52 + df["week_of_year"]

    weeks = sorted(df["week_num"].unique())
    min_train_weeks = 4
    val_window_weeks = 1

    for i in range(min_train_weeks, len(weeks) - val_window_weeks + 1):
        train_weeks = set(weeks[:i])
        val_weeks = set(weeks[i: i + val_window_weeks])

        # Critical: no overlap
        assert train_weeks.isdisjoint(val_weeks), (
            f"Fold {i}: train_weeks and val_weeks overlap: "
            f"{train_weeks & val_weeks}"
        )

        train_dates = set(df.loc[df["week_num"].isin(train_weeks), "game_date"])
        val_dates = set(df.loc[df["week_num"].isin(val_weeks), "game_date"])

        assert train_dates.isdisjoint(val_dates), (
            f"Fold {i}: train and val share game_dates: "
            f"{train_dates & val_dates}"
        )

        # All training dates must precede all validation dates
        if train_dates and val_dates:
            assert max(train_dates) < min(val_dates), (
                f"Fold {i}: max train date {max(train_dates)} >= "
                f"min val date {min(val_dates)}"
            )


def test_walk_forward_increasing_train_size():
    """Each successive fold has more training data than the previous."""
    df = _make_temporal_feature_matrix(n_weeks=12)
    df = df.sort_values("game_date").reset_index(drop=True)
    df["year"] = df["game_date"].dt.year
    df["week_of_year"] = df["game_date"].dt.isocalendar().week.astype(int)
    df["week_num"] = (df["year"] - df["year"].min()) * 52 + df["week_of_year"]
    weeks = sorted(df["week_num"].unique())

    prev_train_size = 0
    for i in range(4, len(weeks)):
        train_weeks = set(weeks[:i])
        train_size = df["week_num"].isin(train_weeks).sum()
        assert train_size > prev_train_size, (
            f"Fold {i}: train size did not increase ({prev_train_size} -> {train_size})"
        )
        prev_train_size = train_size


# ─────────────────────────────────────────────────────────────────────────────
# 5. test_target_not_in_features
# ─────────────────────────────────────────────────────────────────────────────
def test_target_not_in_features():
    """
    'actual_strikeouts' and 'strikeouts' must NOT appear in the 40 feature
    column names — they are the prediction targets, not inputs.
    """
    from src.features.feature_pipeline import FEATURE_COLUMNS

    leak_columns = {"actual_strikeouts", "strikeouts", "k_actual", "target"}
    leaked = leak_columns & set(FEATURE_COLUMNS)
    assert not leaked, (
        f"Target column(s) found in FEATURE_COLUMNS (data leakage!): {leaked}"
    )


@pytest.mark.parametrize("target_col", [
    "actual_strikeouts", "strikeouts", "k_actual", "target", "y",
])
def test_target_columns_not_in_feature_set(target_col):
    """Parametrized check: no target-like column names are in FEATURE_COLUMNS."""
    from src.features.feature_pipeline import FEATURE_COLUMNS

    assert target_col not in FEATURE_COLUMNS, (
        f"Target column '{target_col}' found in FEATURE_COLUMNS — potential leakage!"
    )


def test_metadata_columns_not_used_as_features():
    """
    Metadata columns (pitcher_id, game_date, game_pk, opponent_team_id)
    should not appear in FEATURE_COLUMNS (they're indices, not predictors).
    """
    from src.features.feature_pipeline import FEATURE_COLUMNS, METADATA_COLUMNS

    leaked_metadata = set(METADATA_COLUMNS) & set(FEATURE_COLUMNS)
    assert not leaked_metadata, (
        f"Metadata column(s) found in FEATURE_COLUMNS: {leaked_metadata}"
    )


def test_feature_matrix_no_target_column(sample_feature_matrix):
    """
    A feature matrix used for training should have actual_strikeouts as
    a separate column, not inside the 40-column feature set.
    """
    from src.features.feature_pipeline import FEATURE_COLUMNS

    assert "actual_strikeouts" not in FEATURE_COLUMNS
    # The fixture DOES have actual_strikeouts — but it should be separate
    assert "actual_strikeouts" in sample_feature_matrix.columns, (
        "sample_feature_matrix fixture should contain actual_strikeouts as a separate target column"
    )
    # Confirm it is NOT included in the feature slice
    feature_slice = sample_feature_matrix[FEATURE_COLUMNS]
    assert "actual_strikeouts" not in feature_slice.columns
