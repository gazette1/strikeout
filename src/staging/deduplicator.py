"""
Idempotent deduplication for staging tables.
Ensures append operations never create duplicate rows.
"""
from pathlib import Path
from loguru import logger
import pandas as pd


def deduplicate_parquet(path: Path, key_columns: list[str]) -> int:
    """
    Read a Parquet file, drop duplicates by key columns, rewrite.
    
    Returns:
        Number of duplicate rows removed.
    """
    if not path.exists():
        return 0
    
    df = pd.read_parquet(path)
    original_len = len(df)
    
    available_keys = [c for c in key_columns if c in df.columns]
    if not available_keys:
        logger.warning(f"No key columns found in {path.name}")
        return 0
    
    df = df.drop_duplicates(subset=available_keys, keep="first")
    removed = original_len - len(df)
    
    if removed > 0:
        df.to_parquet(path, index=False)
        logger.info(f"Deduped {path.name}: removed {removed} duplicates ({len(df)} remaining)")
    
    return removed


def deduplicate_staging():
    """Run deduplication across all staging tables."""
    from config.settings import STAGING_PITCHES, STAGING_GAMES
    
    total = 0
    total += deduplicate_parquet(STAGING_PITCHES, ["game_pk", "at_bat_number", "pitch_number"])
    total += deduplicate_parquet(STAGING_GAMES, ["game_pk", "pitcher_id"])
    
    logger.info(f"Total duplicates removed: {total}")
    return total
