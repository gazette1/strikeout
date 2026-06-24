"""
Statcast data loader via pybaseball.
Handles pitch-level data pulls with rate limiting and Parquet caching.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pandas as pd

# Import settings
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_STATCAST, STATCAST_DELAY_SECONDS


def load_statcast_day(date: str, cache: bool = True) -> pd.DataFrame:
    """
    Pull Statcast pitch-level data for a single day.
    
    Args:
        date: Date string 'YYYY-MM-DD'
        cache: If True, read from / write to Parquet cache
    
    Returns:
        DataFrame with all Statcast columns for that day's pitches
    """
    cache_path = RAW_STATCAST / f"{date}.parquet"
    
    if cache and cache_path.exists():
        logger.info(f"Loading cached Statcast data for {date}")
        return pd.read_parquet(cache_path)
    
    logger.info(f"Pulling Statcast data for {date} from Baseball Savant")
    from pybaseball import statcast
    df = statcast(start_dt=date, end_dt=date)
    
    if df is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(df)} pitches for {date}")
    else:
        logger.warning(f"No Statcast data returned for {date}")
        df = pd.DataFrame()
    
    return df


def load_statcast_range(start_date: str, end_date: str,
                        chunk_days: int = 7) -> pd.DataFrame:
    """
    Pull Statcast data for a date range in weekly chunks with rate limiting.
    This is the backfill function.
    
    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        chunk_days: Days per request chunk (default 7)
    
    Returns:
        Combined DataFrame for the full range
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    all_frames = []
    current = start
    
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")
        
        # Check if all days in chunk are cached
        all_cached = True
        chunk_frames = []
        d = current
        while d <= chunk_end:
            day_str = d.strftime("%Y-%m-%d")
            cache_path = RAW_STATCAST / f"{day_str}.parquet"
            if cache_path.exists():
                chunk_frames.append(pd.read_parquet(cache_path))
            else:
                all_cached = False
                break
            d += timedelta(days=1)
        
        if all_cached and chunk_frames:
            logger.info(f"All days cached for {chunk_start_str} to {chunk_end_str}")
            all_frames.extend(chunk_frames)
        else:
            logger.info(f"Pulling Statcast: {chunk_start_str} to {chunk_end_str}")
            from pybaseball import statcast
            df = statcast(start_dt=chunk_start_str, end_dt=chunk_end_str)
            
            if df is not None and not df.empty:
                # Cache each day separately
                if 'game_date' in df.columns:
                    df['game_date'] = pd.to_datetime(df['game_date'])
                    for day_val, day_df in df.groupby(df['game_date'].dt.date):
                        day_str = str(day_val)
                        cache_path = RAW_STATCAST / f"{day_str}.parquet"
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        day_df.to_parquet(cache_path, index=False)
                
                all_frames.append(df)
                logger.info(f"Got {len(df)} pitches for {chunk_start_str} to {chunk_end_str}")
            else:
                logger.warning(f"No data for {chunk_start_str} to {chunk_end_str}")
            
            # Rate limit
            time.sleep(STATCAST_DELAY_SECONDS)
        
        current = chunk_end + timedelta(days=1)
    
    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    return pd.DataFrame()


def load_pitcher_arsenal_stats(year: int, min_pa: int = 25) -> pd.DataFrame:
    """Load arsenal-level outcome stats (whiff%, run value) for a season."""
    from pybaseball import statcast_pitcher_arsenal_stats
    logger.info(f"Loading pitcher arsenal stats for {year}")
    try:
        df = statcast_pitcher_arsenal_stats(year, minPA=min_pa)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to load arsenal stats for {year}: {e}")
        return pd.DataFrame()


def load_pitch_movement(year: int, pitch_type: str = "FF") -> pd.DataFrame:
    """Load league-wide pitch movement stats for computing deltas."""
    from pybaseball import statcast_pitcher_pitch_movement
    logger.info(f"Loading pitch movement for {year}, type={pitch_type}")
    try:
        df = statcast_pitcher_pitch_movement(year, pitch_type=pitch_type)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to load pitch movement for {year}: {e}")
        return pd.DataFrame()
