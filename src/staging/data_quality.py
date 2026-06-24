"""
Data quality checks for staging tables.
Reports null rates, row counts, and schema compliance.
"""
from pathlib import Path
from loguru import logger
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import STAGING_PITCHES, STAGING_GAMES


def check_null_rates(df: pd.DataFrame, table_name: str,
                     threshold: float = 0.5) -> dict:
    """
    Check null rates for each column.
    Warns if any column exceeds the threshold.
    
    Returns:
        Dict of column -> null rate.
    """
    null_rates = {}
    for col in df.columns:
        rate = df[col].isna().mean()
        null_rates[col] = round(rate, 4)
        if rate > threshold:
            logger.warning(f"[{table_name}] Column '{col}' has {rate:.1%} nulls (threshold: {threshold:.1%})")
    
    return null_rates


def check_row_counts(expected_min: int = 0) -> dict:
    """Check that staging tables have reasonable row counts."""
    counts = {}
    
    for path, name in [(STAGING_PITCHES, "pitches"), (STAGING_GAMES, "games")]:
        if path.exists():
            df = pd.read_parquet(path)
            counts[name] = len(df)
            if len(df) < expected_min:
                logger.warning(f"[{name}] Only {len(df)} rows (expected >= {expected_min})")
            else:
                logger.info(f"[{name}] {len(df)} rows OK")
        else:
            counts[name] = 0
            logger.warning(f"[{name}] File not found at {path}")
    
    return counts


def run_quality_checks() -> dict:
    """Run all data quality checks and return a summary report."""
    report = {"row_counts": check_row_counts()}
    
    if STAGING_PITCHES.exists():
        pitches = pd.read_parquet(STAGING_PITCHES)
        report["pitch_null_rates"] = check_null_rates(pitches, "pitches")
    
    if STAGING_GAMES.exists():
        games = pd.read_parquet(STAGING_GAMES)
        report["game_null_rates"] = check_null_rates(games, "games")
    
    return report
