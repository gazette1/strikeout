"""
FanGraphs data loader via pybaseball wrappers.
Loads park factors and team-level batting stats.
"""
from pathlib import Path
from loguru import logger
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROJECT_ROOT, LEAGUE_AVG


def load_team_batting_stats(season: int) -> pd.DataFrame:
    """
    Load team-level batting stats from FanGraphs.
    Includes K%, O-Swing%, Z-Contact%, Contact%, SwStr%.
    """
    logger.info(f"Loading team batting stats for {season}")
    try:
        from pybaseball import team_batting
        df = team_batting(season, season)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to load team batting stats: {e}")
        return pd.DataFrame()


def load_player_batting_stats(season: int) -> pd.DataFrame:
    """Load player-level batting stats with plate discipline metrics."""
    logger.info(f"Loading player batting stats for {season}")
    try:
        from pybaseball import batting_stats
        df = batting_stats(season, season)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to load player batting stats: {e}")
        return pd.DataFrame()


def load_park_factors() -> dict[str, float]:
    """
    Load park K-factors from ballparks.yaml config.
    Returns dict mapping venue_id -> K factor (100 = neutral).
    """
    bp_path = PROJECT_ROOT / "config" / "ballparks.yaml"
    with open(bp_path) as f:
        data = yaml.safe_load(f)
    
    factors = {}
    parks = data.get("ballparks", data)
    if isinstance(parks, list):
        for bp in parks:
            vid = str(bp.get("venue_id", ""))
            factors[vid] = bp.get("park_k_factor", LEAGUE_AVG["park_k_factor"])
    
    return factors


def get_team_k_rate(team_batting_df: pd.DataFrame, team_name: str,
                    pitcher_hand: str = None) -> float:
    """
    Get a team's K% from FanGraphs team batting data.
    If pitcher_hand split is unavailable, returns overall K%.
    """
    if team_batting_df.empty:
        return LEAGUE_AVG["team_k_pct"]
    
    # FanGraphs team batting has a 'Team' column and 'K%' column
    team_row = team_batting_df[
        team_batting_df["Team"].str.contains(team_name, case=False, na=False)
    ]
    
    if team_row.empty:
        return LEAGUE_AVG["team_k_pct"]
    
    k_pct_col = "K%"
    if k_pct_col in team_row.columns:
        val = team_row[k_pct_col].iloc[0]
        if isinstance(val, str):
            val = float(val.strip().replace("%", "")) / 100
        return float(val)
    
    return LEAGUE_AVG["team_k_pct"]
