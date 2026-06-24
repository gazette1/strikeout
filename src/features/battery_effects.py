"""
Battery Effects features (37-40).
Pitcher-catcher pairing effects on strikeout outcomes.
"""
import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUE_AVG, HIGH_ZONE_IDS


def _get_battery_game_logs(game_logs: pd.DataFrame,
                            pitcher_id: int,
                            catcher_id: int,
                            as_of_date: str) -> pd.DataFrame:
    """
    Return game logs where this pitcher-catcher battery appeared together,
    before as_of_date.
    """
    if game_logs is None or (isinstance(game_logs, pd.DataFrame) and game_logs.empty):
        return pd.DataFrame()

    date_mask = pd.to_datetime(game_logs["game_date"]) < pd.to_datetime(as_of_date)
    pitcher_mask = game_logs["pitcher_id"] == pitcher_id

    catcher_col = None
    for col in ("catcher_id", "catcher_mlbam_id", "fielder_2"):
        if col in game_logs.columns:
            catcher_col = col
            break

    if catcher_col is None:
        return pd.DataFrame()

    catcher_mask = game_logs[catcher_col] == catcher_id
    return game_logs.loc[date_mask & pitcher_mask & catcher_mask].copy()


def _pitcher_solo_game_logs(game_logs: pd.DataFrame,
                             pitcher_id: int,
                             catcher_id: int,
                             as_of_date: str) -> pd.DataFrame:
    """Return game logs for pitcher WITHOUT this catcher, before as_of_date."""
    if game_logs is None or (isinstance(game_logs, pd.DataFrame) and game_logs.empty):
        return pd.DataFrame()

    date_mask = pd.to_datetime(game_logs["game_date"]) < pd.to_datetime(as_of_date)
    pitcher_mask = game_logs["pitcher_id"] == pitcher_id

    catcher_col = None
    for col in ("catcher_id", "catcher_mlbam_id", "fielder_2"):
        if col in game_logs.columns:
            catcher_col = col
            break

    if catcher_col is None:
        return game_logs.loc[date_mask & pitcher_mask].copy()

    not_this_catcher = game_logs[catcher_col] != catcher_id
    return game_logs.loc[date_mask & pitcher_mask & not_this_catcher].copy()


def _k9_from_game_logs(gl: pd.DataFrame) -> float:
    """Compute K/9 from a game_logs slice. Returns NaN if empty."""
    if gl.empty:
        return np.nan
    total_ip = gl["innings_pitched"].clip(lower=0.1).sum()
    total_k = gl["strikeouts"].sum()
    if total_ip == 0:
        return np.nan
    return float((total_k / total_ip) * 9)


# ── Feature 37: Catcher framing runs ─────────────────────────
def feat_catcher_framing_runs(catcher_id: int = 0) -> float:
    """
    Catcher framing runs above average (Statcast-derived).
    Returns NaN — let LightGBM handle natively per blueprint.
    Blueprint Feature 37.

    Stub: When framing data is ingested, load from staging/framing.parquet
    and compute runs_above_avg for the catcher over the current season.
    """
    # Intentionally return NaN for native LightGBM handling
    return np.nan


# ── Feature 38: Battery K rate together ───────────────────────
def feat_battery_k_rate_together(game_logs: pd.DataFrame,
                                  lineups: pd.DataFrame,
                                  pitcher_id: int,
                                  catcher_id: int,
                                  as_of_date: str) -> float:
    """
    K/9 when this pitcher-catcher battery has appeared together.
    Falls back to pitcher overall K/9 if fewer than 3 paired starts.
    Blueprint Feature 38.
    """
    if catcher_id == 0 or catcher_id is None:
        return np.nan

    battery_gl = _get_battery_game_logs(game_logs, pitcher_id, catcher_id, as_of_date)

    if len(battery_gl) < 3:
        # Not enough paired history — fall back to pitcher's overall K/9
        if game_logs is not None and not game_logs.empty:
            pitcher_gl = game_logs[
                (game_logs["pitcher_id"] == pitcher_id) &
                (pd.to_datetime(game_logs["game_date"]) < pd.to_datetime(as_of_date))
            ]
            return _k9_from_game_logs(pitcher_gl)
        return LEAGUE_AVG["k_per_9"]

    return _k9_from_game_logs(battery_gl)


# ── Feature 39: Battery K rate delta ─────────────────────────
def feat_battery_k_rate_delta(game_logs: pd.DataFrame,
                               pitcher_id: int,
                               catcher_id: int,
                               as_of_date: str) -> float:
    """
    Delta = K/9 together - K/9 apart (pitcher without this catcher).
    Returns NaN if insufficient data (native LightGBM).
    Blueprint Feature 39.
    """
    if catcher_id == 0 or catcher_id is None:
        return np.nan

    battery_gl = _get_battery_game_logs(game_logs, pitcher_id, catcher_id, as_of_date)
    if len(battery_gl) < 3:
        return np.nan

    apart_gl = _pitcher_solo_game_logs(game_logs, pitcher_id, catcher_id, as_of_date)
    if len(apart_gl) < 3:
        return np.nan

    k9_together = _k9_from_game_logs(battery_gl)
    k9_apart = _k9_from_game_logs(apart_gl)

    if pd.isna(k9_together) or pd.isna(k9_apart):
        return np.nan

    return float(k9_together - k9_apart)


# ── Feature 40: Catcher game-calling aggression ───────────────
def feat_catcher_game_calling_aggression(pitches: pd.DataFrame,
                                          catcher_id: int,
                                          as_of_date: str,
                                          lookback_games: int = 30) -> float:
    """
    Ratio of high-zone called pitches (HIGH_ZONE_IDS) to total called pitches.
    High aggression = catcher calling more elevated pitches (favors K).
    Returns NaN if catcher_id not found (native LightGBM).
    Blueprint Feature 40.
    """
    if catcher_id == 0 or catcher_id is None:
        return np.nan

    if pitches is None or (isinstance(pitches, pd.DataFrame) and pitches.empty):
        return np.nan

    # Find catcher column in Statcast data
    catcher_col = None
    for col in ("fielder_2", "catcher_id", "fielder_2_1"):
        if col in pitches.columns:
            catcher_col = col
            break

    if catcher_col is None:
        return np.nan

    date_mask = pd.to_datetime(pitches["game_date"]) < pd.to_datetime(as_of_date)
    catcher_mask = pitches[catcher_col] == catcher_id

    catcher_pitches = pitches.loc[date_mask & catcher_mask].copy()
    if catcher_pitches.empty:
        return np.nan

    # Limit to last N games
    recent_games = sorted(catcher_pitches["game_date"].unique(), reverse=True)[:lookback_games]
    catcher_pitches = catcher_pitches[catcher_pitches["game_date"].isin(recent_games)]

    if catcher_pitches.empty or "zone" not in catcher_pitches.columns:
        return np.nan

    # Called pitches = pitches where the catcher received (all pitches)
    total_pitches = len(catcher_pitches)
    if total_pitches < 50:
        return np.nan

    high_zone_count = catcher_pitches["zone"].isin(HIGH_ZONE_IDS).sum()
    return float(high_zone_count / total_pitches)


def compute_all_battery_effects(pitches: pd.DataFrame,
                                  game_logs: pd.DataFrame,
                                  lineups: pd.DataFrame,
                                  pitcher_id: int,
                                  catcher_id: int,
                                  as_of_date: str) -> dict:
    """
    Compute all 4 battery effect features.
    Returns NaN for unavailable catcher data (native LightGBM).
    """
    return {
        "catcher_framing_runs": feat_catcher_framing_runs(catcher_id),
        "battery_k_rate_together": feat_battery_k_rate_together(
            game_logs, lineups, pitcher_id, catcher_id, as_of_date),
        "battery_k_rate_delta": feat_battery_k_rate_delta(
            game_logs, pitcher_id, catcher_id, as_of_date),
        "catcher_game_calling_aggression": feat_catcher_game_calling_aggression(
            pitches, catcher_id, as_of_date),
    }
