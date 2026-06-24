"""
Historical data backfill script.
Run once to pull 2022-2025 Statcast + MLB Stats API data.

Usage:
    python -m src.data.backfill --start-year 2022 --end-year 2025
    python -m src.data.backfill --start-year 2024 --end-year 2024 --start-month 7 --end-month 7
"""
import argparse
from datetime import datetime, timedelta
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.statcast_loader import load_statcast_range
from src.data.mlb_api_loader import extract_pitcher_game_logs, extract_lineups
from src.staging.schema_validator import validate_and_stage_pitches, validate_and_stage_games


# MLB regular season approximate date ranges
SEASON_DATES = {
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}


def backfill_statcast(start_year: int, end_year: int,
                      start_month: int = None, end_month: int = None):
    """Pull Statcast data for specified year range."""
    for year in range(start_year, end_year + 1):
        season_start, season_end = SEASON_DATES.get(year, (f"{year}-03-28", f"{year}-09-29"))
        
        if start_month:
            season_start = f"{year}-{start_month:02d}-01"
        if end_month:
            # Last day of the month
            if end_month == 12:
                season_end = f"{year}-12-31"
            else:
                season_end = f"{year}-{end_month + 1:02d}-01"
                season_end = (datetime.strptime(season_end, "%Y-%m-%d") -
                              timedelta(days=1)).strftime("%Y-%m-%d")
        
        logger.info(f"=== Backfilling Statcast for {year}: {season_start} to {season_end} ===")
        df = load_statcast_range(season_start, season_end)
        logger.info(f"Statcast {year}: {len(df)} total pitches loaded")
        
        # Stage the data
        if not df.empty:
            validate_and_stage_pitches(df)


def backfill_game_logs(start_year: int, end_year: int,
                       start_month: int = None, end_month: int = None):
    """Pull game logs for specified year range."""
    import pandas as pd

    for year in range(start_year, end_year + 1):
        season_start, season_end = SEASON_DATES.get(year, (f"{year}-03-28", f"{year}-09-29"))
        
        if start_month:
            season_start = f"{year}-{start_month:02d}-01"
        if end_month:
            if end_month == 12:
                season_end = f"{year}-12-31"
            else:
                next_month = f"{year}-{end_month + 1:02d}-01"
                season_end = (datetime.strptime(next_month, "%Y-%m-%d") -
                              timedelta(days=1)).strftime("%Y-%m-%d")
        
        logger.info(f"=== Backfilling game logs for {year}: {season_start} to {season_end} ===")
        
        current = datetime.strptime(season_start, "%Y-%m-%d")
        end = datetime.strptime(season_end, "%Y-%m-%d")
        total_logs = 0
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                gl = extract_pitcher_game_logs(date_str)
                lu = extract_lineups(date_str)
                total_logs += len(gl)
            except Exception as e:
                logger.warning(f"Failed to load game logs for {date_str}: {e}")
            current += timedelta(days=1)
        
        logger.info(f"Game logs {year}: {total_logs} pitcher starts loaded")
        
        # Stage the game logs
        validate_and_stage_games(year)


def main():
    parser = argparse.ArgumentParser(description="MLB K-Predictor Historical Backfill")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=None)
    parser.add_argument("--end-month", type=int, default=None)
    parser.add_argument("--skip-statcast", action="store_true")
    parser.add_argument("--skip-game-logs", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"Starting backfill: {args.start_year}-{args.end_year}")
    
    if not args.skip_statcast:
        backfill_statcast(args.start_year, args.end_year, args.start_month, args.end_month)
    
    if not args.skip_game_logs:
        backfill_game_logs(args.start_year, args.end_year, args.start_month, args.end_month)
    
    logger.info("Backfill complete!")


if __name__ == "__main__":
    main()
