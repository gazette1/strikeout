"""
Contextual features (32-36).
Park factors, umpire tendencies, weather conditions, and bullpen leash.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUE_AVG

# ── Park factors lookup ───────────────────────────────────────
_BALLPARK_K_FACTORS: dict = {}
_BALLPARKS_LOADED: bool = False


def _load_ballpark_factors() -> dict:
    """
    Load park K factors from config/ballparks.yaml.
    Returns a dict of team_abbr -> k_factor (index 100 = neutral).
    """
    global _BALLPARK_K_FACTORS, _BALLPARKS_LOADED

    if _BALLPARKS_LOADED:
        return _BALLPARK_K_FACTORS

    try:
        import yaml
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "ballparks.yaml"
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        # Build factor lookup from team abbreviation
        # ballparks.yaml includes team, venue, elevation_m, is_dome, etc.
        # We derive K factors heuristically from elevation and dome status:
        #   - Higher elevation -> more air resistance -> slightly fewer K (factor <100)
        #   - Dome parks -> controlled environment -> neutral (100)
        #   - Sea-level open air -> slight positive
        factors = {}
        for park in data.get("ballparks", []):
            team = park.get("team", "")
            elevation = park.get("elevation_m", 0)
            is_dome = park.get("is_dome", False)

            if is_dome:
                factor = 100
            elif elevation > 1000:
                # High altitude (Coors Field) suppresses K slightly
                factor = 96
            elif elevation > 200:
                factor = 99
            else:
                factor = 101

            factors[team] = factor

        _BALLPARK_K_FACTORS = factors
        _BALLPARKS_LOADED = True
        logger.debug(f"Loaded K factors for {len(factors)} ballparks")
    except Exception as exc:
        logger.warning(f"Could not load ballparks.yaml: {exc}; using neutral factors")
        _BALLPARK_K_FACTORS = {}
        _BALLPARKS_LOADED = True

    return _BALLPARK_K_FACTORS


# ── Feature 32: Park K factor ─────────────────────────────────
def feat_park_k_factor(ballpark_id: str) -> float:
    """
    Park-level strikeout factor (index, 100 = league neutral).
    Loaded from config/ballparks.yaml; fallback to 100.
    Blueprint Feature 32.
    """
    factors = _load_ballpark_factors()

    if not ballpark_id:
        return float(LEAGUE_AVG["park_k_factor"])

    # ballpark_id may be a team abbreviation or venue name
    factor = factors.get(str(ballpark_id).upper())
    if factor is None:
        # Try case-insensitive partial match on team key
        key = str(ballpark_id).upper()
        for team, val in factors.items():
            if key in team or team in key:
                factor = val
                break

    return float(factor) if factor is not None else float(LEAGUE_AVG["park_k_factor"])


# ── Feature 33: Umpire K boost ────────────────────────────────
def feat_ump_k_boost(umpire_history,
                      umpire_id: int,
                      as_of_date: str) -> float:
    """
    Umpire's historical K-rate boost vs league average.
    Delegates to umpire_loader.compute_ump_k_boost.
    Returns NaN if umpire not found or < 20 games (native LightGBM NaN).
    Blueprint Feature 33.
    """
    if umpire_history is None or umpire_id == 0:
        return np.nan

    try:
        from src.data.umpire_loader import compute_ump_k_boost
        boost = compute_ump_k_boost(umpire_history, umpire_id, as_of_date)
        return float(boost) if boost is not None and not pd.isna(boost) else np.nan
    except (ImportError, AttributeError):
        # umpire_loader not yet implemented or missing function; fall through
        pass
    except Exception as exc:
        logger.debug(f"ump_k_boost lookup failed for umpire {umpire_id}: {exc}")

    # Direct lookup if umpire_history is a DataFrame/dict
    if isinstance(umpire_history, pd.DataFrame):
        if umpire_history.empty:
            return np.nan
        ump_mask = umpire_history.get(
            "umpire_id", pd.Series(dtype=int)) == umpire_id
        ump_rows = umpire_history.loc[ump_mask]

        date_filtered = ump_rows[
            pd.to_datetime(ump_rows.get("game_date", pd.Series(dtype=str))) <
            pd.to_datetime(as_of_date)
        ] if "game_date" in ump_rows.columns else ump_rows

        if len(date_filtered) < 20:
            return np.nan
        if "k_boost" in date_filtered.columns:
            return float(date_filtered["k_boost"].mean())
        if "k_pct" in date_filtered.columns:
            return float(date_filtered["k_pct"].mean() - LEAGUE_AVG.get("team_k_pct", 0.225))

    elif isinstance(umpire_history, dict):
        boost = umpire_history.get(umpire_id)
        if boost is not None:
            return float(boost)

    return np.nan


# ── Feature 34: Temperature (°F) ──────────────────────────────
def feat_temperature_f(weather_df: pd.DataFrame, game_pk) -> float:
    """
    Game-time temperature in Fahrenheit.
    Fallback to 72°F (dome default).
    Blueprint Feature 34.
    """
    if weather_df is None or (isinstance(weather_df, pd.DataFrame) and weather_df.empty):
        return float(LEAGUE_AVG["dome_temp_f"])

    if not isinstance(weather_df, pd.DataFrame):
        return float(LEAGUE_AVG["dome_temp_f"])

    mask = weather_df.get("game_pk", pd.Series(dtype=int)) == game_pk
    game_weather = weather_df.loc[mask]

    if game_weather.empty:
        return float(LEAGUE_AVG["dome_temp_f"])

    for col in ("temperature_f", "temp_f", "temperature", "temp"):
        if col in game_weather.columns:
            val = game_weather[col].iloc[0]
            if pd.notna(val):
                # Sanity clamp: 20°F to 110°F
                return float(max(20.0, min(110.0, float(val))))

    return float(LEAGUE_AVG["dome_temp_f"])


# ── Feature 35: Humidity % ────────────────────────────────────
def feat_humidity_pct(weather_df: pd.DataFrame, game_pk) -> float:
    """
    Game-time relative humidity percentage (0-100).
    Fallback to 50% (dome default).
    Blueprint Feature 35.
    """
    if weather_df is None or (isinstance(weather_df, pd.DataFrame) and weather_df.empty):
        return float(LEAGUE_AVG["dome_humidity_pct"])

    if not isinstance(weather_df, pd.DataFrame):
        return float(LEAGUE_AVG["dome_humidity_pct"])

    mask = weather_df.get("game_pk", pd.Series(dtype=int)) == game_pk
    game_weather = weather_df.loc[mask]

    if game_weather.empty:
        return float(LEAGUE_AVG["dome_humidity_pct"])

    for col in ("humidity_pct", "humidity", "relative_humidity"):
        if col in game_weather.columns:
            val = game_weather[col].iloc[0]
            if pd.notna(val):
                return float(max(0.0, min(100.0, float(val))))

    return float(LEAGUE_AVG["dome_humidity_pct"])


# ── Feature 36: Pitcher leash ─────────────────────────────────
def feat_pitcher_leash(game_logs: pd.DataFrame,
                        team_id,
                        as_of_date: str,
                        lookback_days: int = 2) -> float:
    """
    Ratio of bullpen pitches thrown in last N days to team average.
    > 1.0 = bullpen overused -> pitcher may have shorter leash.
    Blueprint Feature 36.
    """
    if game_logs is None or (isinstance(game_logs, pd.DataFrame) and game_logs.empty):
        return 1.0  # Neutral

    if not isinstance(game_logs, pd.DataFrame):
        return 1.0

    as_of = pd.to_datetime(as_of_date)
    cutoff = as_of - pd.Timedelta(days=lookback_days)

    # Filter to team games
    team_col = None
    for col in ("team_id", "home_team", "away_team"):
        if col in game_logs.columns:
            team_col = col
            break

    if team_col is None:
        return 1.0

    team_mask = game_logs[team_col] == team_id
    date_mask = pd.to_datetime(game_logs["game_date"]) < as_of

    team_games = game_logs.loc[team_mask & date_mask]

    if team_games.empty:
        return 1.0

    # Bullpen pitches column
    bullpen_col = None
    for col in ("bullpen_pitches", "relief_pitches", "rp_pitches_thrown"):
        if col in team_games.columns:
            bullpen_col = col
            break

    if bullpen_col is None:
        return 1.0

    # Recent window
    recent_games = team_games[
        pd.to_datetime(team_games["game_date"]) >= cutoff
    ]

    if recent_games.empty:
        return 1.0

    recent_bp = recent_games[bullpen_col].sum()
    avg_bp = team_games[bullpen_col].mean() * lookback_days

    if avg_bp == 0:
        return 1.0

    return float(recent_bp / avg_bp)


def compute_all_contextual(ballpark_id: str,
                            umpire_history,
                            umpire_id: int,
                            as_of_date: str,
                            weather_df: pd.DataFrame,
                            game_pk,
                            game_logs: pd.DataFrame,
                            team_id) -> dict:
    """Compute all 5 contextual features. Returns dict of feature_name -> value."""
    return {
        "park_k_factor": feat_park_k_factor(ballpark_id),
        "ump_k_boost": feat_ump_k_boost(umpire_history, umpire_id, as_of_date),
        "temperature_f": feat_temperature_f(weather_df, game_pk),
        "humidity_pct": feat_humidity_pct(weather_df, game_pk),
        "pitcher_leash": feat_pitcher_leash(game_logs, team_id, as_of_date),
    }
