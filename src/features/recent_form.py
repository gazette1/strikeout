"""
Recent Form features (15-18).
Computed from pitcher game logs — short-window performance indicators.
"""
import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUE_AVG


def _get_pitcher_game_logs(game_logs: pd.DataFrame, pitcher_id: int,
                            as_of_date: str) -> pd.DataFrame:
    """Return game logs for a pitcher before as_of_date, sorted oldest->newest."""
    mask = (
        (game_logs["pitcher_id"] == pitcher_id) &
        (pd.to_datetime(game_logs["game_date"]) < pd.to_datetime(as_of_date))
    )
    return game_logs.loc[mask].sort_values("game_date").copy()


def _get_season_game_logs(game_logs: pd.DataFrame, pitcher_id: int,
                           as_of_date: str) -> pd.DataFrame:
    """Return game logs for a pitcher in the current season before as_of_date."""
    gl = _get_pitcher_game_logs(game_logs, pitcher_id, as_of_date)
    if gl.empty:
        return gl
    year = pd.to_datetime(as_of_date).year
    return gl[pd.to_datetime(gl["game_date"]).dt.year == year]


def _get_last_n_starts(game_logs: pd.DataFrame, pitcher_id: int,
                        as_of_date: str, n_starts: int = 5) -> pd.DataFrame:
    """Return last N game-log rows for pitcher before as_of_date."""
    gl = _get_pitcher_game_logs(game_logs, pitcher_id, as_of_date)
    return gl.tail(n_starts)


# ── Feature 15: K/9 over last 5 starts (linear recency weighting) ──
def feat_k_rate_last_5(game_logs: pd.DataFrame, pitcher_id: int,
                        as_of_date: str, n_starts: int = 5) -> float:
    """
    K/9 over last N starts with linear recency weighting.
    Most recent start gets weight=n, oldest gets weight=1.
    Blueprint Feature 15.
    """
    gl = _get_last_n_starts(game_logs, pitcher_id, as_of_date, n_starts)

    if gl.empty:
        # Fall back to season K/9
        season_gl = _get_season_game_logs(game_logs, pitcher_id, as_of_date)
        if season_gl.empty:
            return LEAGUE_AVG["k_per_9"]
        total_ip = season_gl["innings_pitched"].clip(lower=0.1).sum()
        total_k = season_gl["strikeouts"].sum()
        return float((total_k / total_ip) * 9)

    n = len(gl)
    # Linear weights: oldest = 1, most recent = n
    weights = np.arange(1, n + 1, dtype=float)

    k9_values = (gl["strikeouts"].values /
                 gl["innings_pitched"].clip(lower=0.1).values) * 9

    weighted_sum = np.dot(weights, k9_values)
    weight_total = weights.sum()

    return float(weighted_sum / weight_total)


# ── Feature 16: Recent pitch count trend ──────────────────────
def feat_recent_pitch_count_trend(game_logs: pd.DataFrame, pitcher_id: int,
                                   as_of_date: str, n_starts: int = 5) -> float:
    """
    Linear slope of pitches_thrown over last N starts.
    Positive slope = increasing pitch counts (pitcher going deeper).
    Blueprint Feature 16.
    """
    gl = _get_last_n_starts(game_logs, pitcher_id, as_of_date, n_starts)

    if len(gl) < 2:
        return 0.0  # Not enough data; no trend

    x = np.arange(len(gl), dtype=float)
    y = gl["pitches_thrown"].values.astype(float)

    # Handle NaN values in pitches_thrown
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        return 0.0

    slope, _ = np.polyfit(x[valid_mask], y[valid_mask], 1)
    return float(slope)


# ── Feature 17: IP per start (blended) ────────────────────────
def feat_ip_per_start(game_logs: pd.DataFrame, pitcher_id: int,
                       as_of_date: str, n_starts: int = 5,
                       recent_weight: float = 0.6,
                       season_weight: float = 0.4) -> float:
    """
    Blended IP per start: 60% last N starts mean + 40% season mean.
    Fallback to LEAGUE_AVG["ip_per_start"].
    Blueprint Feature 17.
    """
    season_gl = _get_season_game_logs(game_logs, pitcher_id, as_of_date)

    if season_gl.empty:
        return LEAGUE_AVG["ip_per_start"]

    season_mean = season_gl["innings_pitched"].mean()

    recent_gl = season_gl.tail(n_starts)
    recent_mean = recent_gl["innings_pitched"].mean() if len(recent_gl) >= 1 else season_mean

    if pd.isna(season_mean):
        season_mean = LEAGUE_AVG["ip_per_start"]
    if pd.isna(recent_mean):
        recent_mean = season_mean

    blended = recent_weight * recent_mean + season_weight * season_mean
    return float(blended)


# ── Feature 18: Days rest ─────────────────────────────────────
def feat_days_rest(game_logs: pd.DataFrame, pitcher_id: int,
                    as_of_date: str) -> float:
    """
    Days between as_of_date and pitcher's last start.
    Fallback to 5 (standard MLB rotation rest).
    Blueprint Feature 18.
    """
    gl = _get_pitcher_game_logs(game_logs, pitcher_id, as_of_date)

    if gl.empty:
        return float(LEAGUE_AVG["days_rest_default"])

    last_game_date = pd.to_datetime(gl["game_date"].iloc[-1])
    as_of = pd.to_datetime(as_of_date)
    days = (as_of - last_game_date).days

    # Sanity clamp: 1 to 30 days
    if days < 1 or days > 30:
        return float(LEAGUE_AVG["days_rest_default"])

    return float(days)


def compute_all_recent_form(game_logs: pd.DataFrame, pitcher_id: int,
                             as_of_date: str) -> dict:
    """Compute all 4 recent form features. Returns dict of feature_name -> value."""
    return {
        "k_rate_last_5": feat_k_rate_last_5(game_logs, pitcher_id, as_of_date),
        "recent_pitch_count_trend": feat_recent_pitch_count_trend(game_logs, pitcher_id, as_of_date),
        "ip_per_start": feat_ip_per_start(game_logs, pitcher_id, as_of_date),
        "days_rest": feat_days_rest(game_logs, pitcher_id, as_of_date),
    }
