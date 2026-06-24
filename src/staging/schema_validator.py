"""
Schema validation and staging for raw data.
Enforces column types and nullability before appending to staging tables.
"""
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import STAGING_DIR, STAGING_PITCHES, STAGING_GAMES, RAW_GAME_LOGS


# Expected dtypes for staging tables
PITCH_SCHEMA = {
    "game_pk": "int64",
    "game_date": "object",  # Will be converted to datetime
    "pitcher": "int64",
    "batter": "int64",
    "pitch_type": "object",
    "release_speed": "float64",
    "release_spin_rate": "float64",
    "pfx_x": "float64",
    "pfx_z": "float64",
    "release_pos_x": "float64",
    "release_pos_z": "float64",
    "plate_x": "float64",
    "plate_z": "float64",
    "vx0": "float64",
    "vy0": "float64",
    "vz0": "float64",
    "ax": "float64",
    "ay": "float64",
    "az": "float64",
    "zone": "float64",
    "description": "object",
    "events": "object",
    "strikes": "float64",
    "balls": "float64",
    "stand": "object",
    "p_throws": "object",
    "at_bat_number": "float64",
    "pitch_number": "float64",
    "inning": "float64",
}

GAME_LOG_SCHEMA = {
    "game_pk": "int64",
    "game_date": "object",
    "pitcher_id": "int64",
    "pitcher_name": "object",
    "team_id": "int64",
    "opponent_team_id": "int64",
    "is_home": "bool",
    "innings_pitched": "float64",
    "strikeouts": "int64",
    "pitches_thrown": "int64",
    "walks": "int64",
    "earned_runs": "int64",
    "hits_allowed": "int64",
    "home_plate_umpire_id": "int64",
    "home_plate_umpire": "object",
    "ballpark_id": "object",
    "game_time_et": "object",
}


def validate_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce pitch data to expected schema."""
    if df.empty:
        return df
    
    # Keep only columns we use (Statcast returns 118 cols)
    available = [c for c in PITCH_SCHEMA if c in df.columns]
    df = df[available].copy()
    
    # Coerce types
    for col, dtype in PITCH_SCHEMA.items():
        if col in df.columns:
            if dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Can't have int with NaN, keep as float
            elif dtype == "object":
                df[col] = df[col].astype(str).replace("nan", pd.NA)
    
    # Convert game_date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    
    # Drop complete duplicates
    key_cols = ["game_pk", "at_bat_number", "pitch_number"]
    key_available = [c for c in key_cols if c in df.columns]
    if key_available:
        df = df.drop_duplicates(subset=key_available, keep="first")
    
    return df


def validate_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce game log data."""
    if df.empty:
        return df
    
    for col, dtype in GAME_LOG_SCHEMA.items():
        if col in df.columns:
            if dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "bool":
                df[col] = df[col].astype(bool)
    
    # Deduplicate by primary key
    df = df.drop_duplicates(subset=["game_pk", "pitcher_id"], keep="first")
    return df


def validate_and_stage_pitches(new_df: pd.DataFrame) -> None:
    """Validate new pitch data and append to staging."""
    validated = validate_pitches(new_df)
    if validated.empty:
        return
    
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    if STAGING_PITCHES.exists():
        existing = pd.read_parquet(STAGING_PITCHES)
        combined = pd.concat([existing, validated], ignore_index=True)
        key_cols = ["game_pk", "at_bat_number", "pitch_number"]
        key_available = [c for c in key_cols if c in combined.columns]
        if key_available:
            combined = combined.drop_duplicates(subset=key_available, keep="first")
    else:
        combined = validated
    
    combined.to_parquet(STAGING_PITCHES, index=False)
    logger.info(f"Staging pitches: {len(combined)} total rows")


def validate_and_stage_games(year: int = None) -> None:
    """Load all raw game logs and stage them."""
    from config.settings import RAW_GAME_LOGS
    
    frames = []
    for f in sorted(RAW_GAME_LOGS.glob("*.parquet")):
        df = pd.read_parquet(f)
        if year and "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])
            df = df[df["game_date"].dt.year == year]
        frames.append(df)
    
    if not frames:
        return
    
    combined = pd.concat(frames, ignore_index=True)
    combined = validate_game_logs(combined)
    
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    if STAGING_GAMES.exists():
        existing = pd.read_parquet(STAGING_GAMES)
        combined = pd.concat([existing, combined], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_pk", "pitcher_id"], keep="first")
    
    combined.to_parquet(STAGING_GAMES, index=False)
    logger.info(f"Staging games: {len(combined)} total rows")
