"""
Umpire scorecard data loader.
Uses Kaggle historical dataset + derives K boost from Statcast called strikes.
"""
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_UMPIRES, STAGING_DIR, LEAGUE_AVG


def load_umpire_history() -> pd.DataFrame:
    """
    Load the umpire history table from staging.
    If it doesn't exist, return empty DataFrame with expected schema.
    """
    path = STAGING_DIR / "umpire_history.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[
        "umpire_id", "umpire_name", "game_date", "game_pk",
        "pitches_called", "called_strikes", "incorrect_calls",
        "called_strike_rate", "k_boost_score",
    ])


def build_umpire_stats_from_statcast(pitches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-umpire called-strike rates from Statcast pitch-level data.
    This is used when UmpScorecards data is unavailable.
    
    Groups by (game_pk, home_plate_umpire) and computes:
    - pitches_called: count of called balls + called strikes
    - called_strikes: count of 'called_strike' descriptions
    - called_strike_rate: called_strikes / pitches_called
    """
    if pitches_df.empty:
        return pd.DataFrame()
    
    called_descs = {"called_strike", "ball", "blocked_ball"}
    called = pitches_df[pitches_df["description"].isin(called_descs)].copy()
    
    if called.empty:
        return pd.DataFrame()
    
    called["is_called_strike"] = (called["description"] == "called_strike").astype(int)
    
    # We need umpire info joined from game_logs
    # For now, return per-game aggregation that can be joined later
    agg = called.groupby(["game_pk", "game_date"]).agg(
        pitches_called=("is_called_strike", "count"),
        called_strikes=("is_called_strike", "sum"),
    ).reset_index()
    
    agg["called_strike_rate"] = agg["called_strikes"] / agg["pitches_called"].clip(lower=1)
    return agg


def compute_ump_k_boost(umpire_history: pd.DataFrame,
                        umpire_id: int,
                        as_of_date: str,
                        min_games: int = 20) -> float:
    """
    Compute K boost score for an umpire: rolling 2-season average of
    (ump_called_strike_rate - league_avg) * expected_called_pitches_per_game.
    
    Returns 0.0 (neutral) if umpire has fewer than min_games in history
    or is not found.
    """
    if umpire_history.empty:
        return np.nan
    
    as_of = pd.to_datetime(as_of_date)
    # Filter to last 2 seasons (roughly 2 years) before the prediction date
    cutoff = as_of - pd.DateOffset(years=2)
    
    ump_data = umpire_history[
        (umpire_history["umpire_id"] == umpire_id) &
        (pd.to_datetime(umpire_history["game_date"]) >= cutoff) &
        (pd.to_datetime(umpire_history["game_date"]) < as_of)
    ]
    
    if len(ump_data) < min_games:
        return np.nan  # Let LightGBM handle natively per blueprint
    
    avg_cs_rate = ump_data["called_strike_rate"].mean()
    league_avg_cs_rate = 0.335  # Approximate league average called strike rate
    expected_called_per_game = 75  # Per blueprint
    
    k_boost = (avg_cs_rate - league_avg_cs_rate) * expected_called_per_game
    return k_boost


def save_umpire_history(df: pd.DataFrame) -> None:
    """Save umpire history to staging."""
    path = STAGING_DIR / "umpire_history.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} umpire history records")
