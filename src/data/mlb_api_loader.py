"""
MLB Stats API loader via the statsapi library.
Handles game logs, lineups, schedules, umpire assignments.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_GAME_LOGS, RAW_LINEUPS, MLB_API_DELAY_SECONDS


def _throttle():
    time.sleep(MLB_API_DELAY_SECONDS)


def load_schedule(date: str) -> list[dict]:
    """Get all MLB games scheduled for a date. Returns list of game dicts."""
    import statsapi
    logger.info(f"Loading schedule for {date}")
    games = statsapi.schedule(start_date=date, end_date=date)
    _throttle()
    return games


def load_boxscore(game_pk: int) -> dict:
    """Get full boxscore data for a game."""
    import statsapi
    data = statsapi.boxscore_data(game_pk)
    _throttle()
    return data


def load_game_data(game_pk: int) -> dict:
    """Get live game data (lineups, officials, weather)."""
    import statsapi
    data = statsapi.get("game", {"gamePk": game_pk})
    _throttle()
    return data


def extract_pitcher_game_logs(date: str) -> pd.DataFrame:
    """
    For all games on a date, extract one row per starting pitcher
    with: game_pk, pitcher_id, pitcher_name, team_id, opponent_team_id,
    is_home, innings_pitched, strikeouts, pitches_thrown, walks,
    earned_runs, hits_allowed, home_plate_umpire, ballpark_id, game_time_et.
    
    Saves to RAW_GAME_LOGS/{date}.parquet.
    """
    cache_path = RAW_GAME_LOGS / f"{date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    games = load_schedule(date)
    if not games:
        logger.warning(f"No games found for {date}")
        return pd.DataFrame()
    
    rows = []
    for game in games:
        game_pk = game["game_id"]
        status = game.get("status", "")
        if "Final" not in status and "Completed" not in status:
            continue
        
        try:
            box = load_boxscore(game_pk)
            game_data = load_game_data(game_pk)
        except Exception as e:
            logger.warning(f"Failed to load game {game_pk}: {e}")
            continue
        
        # Extract umpire
        hp_umpire = ""
        hp_umpire_id = 0
        try:
            officials = game_data.get("liveData", {}).get("boxscore", {}).get("officials", [])
            for official in officials:
                if official.get("officialType") == "Home Plate":
                    hp_umpire = official.get("official", {}).get("fullName", "")
                    hp_umpire_id = official.get("official", {}).get("id", 0)
                    break
        except Exception:
            pass
        
        # Extract venue
        venue_id = ""
        try:
            venue_id = str(game_data.get("gameData", {}).get("venue", {}).get("id", ""))
        except Exception:
            pass
        
        # Extract starting pitchers from both sides
        for side in ["home", "away"]:
            try:
                team_key = f"{side}Pitchers"
                other_side = "away" if side == "home" else "home"
                
                # The first pitcher listed is typically the starter
                pitchers = box.get(team_key, [])
                if not pitchers:
                    continue
                
                # statsapi boxscore_data returns pitcher stats
                starter_data = pitchers[0] if isinstance(pitchers, list) else None
                if starter_data is None:
                    continue
                
                # Parse IP (e.g., "6.1" means 6 and 1/3)
                ip_str = str(starter_data.get("ip", "0.0"))
                ip_parts = ip_str.split(".")
                ip = int(ip_parts[0]) + (int(ip_parts[1]) / 3 if len(ip_parts) > 1 else 0)
                
                rows.append({
                    "game_pk": game_pk,
                    "game_date": date,
                    "pitcher_id": starter_data.get("personId", 0),
                    "pitcher_name": starter_data.get("name", ""),
                    "team_id": game.get(f"{side}_id", 0),
                    "opponent_team_id": game.get(f"{other_side}_id", 0),
                    "is_home": side == "home",
                    "innings_pitched": ip,
                    "strikeouts": int(starter_data.get("k", 0)),
                    "pitches_thrown": int(starter_data.get("p", 0)),
                    "walks": int(starter_data.get("bb", 0)),
                    "earned_runs": int(starter_data.get("er", 0)),
                    "hits_allowed": int(starter_data.get("h", 0)),
                    "home_plate_umpire_id": hp_umpire_id,
                    "home_plate_umpire": hp_umpire,
                    "ballpark_id": venue_id,
                    "game_time_et": game.get("game_datetime", ""),
                })
            except Exception as e:
                logger.warning(f"Error parsing {side} pitcher for game {game_pk}: {e}")
    
    df = pd.DataFrame(rows)
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(df)} pitcher game logs for {date}")
    
    return df


def extract_lineups(date: str) -> pd.DataFrame:
    """
    Extract batting lineups for all games on a date.
    One row per batter: game_pk, team_id, batter_id, batter_name,
    batting_order, position, handedness, is_starter.
    """
    cache_path = RAW_LINEUPS / f"{date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    games = load_schedule(date)
    rows = []
    
    for game in games:
        game_pk = game["game_id"]
        status = game.get("status", "")
        if "Final" not in status and "Completed" not in status:
            continue
        
        try:
            box = load_boxscore(game_pk)
        except Exception as e:
            logger.warning(f"Failed to load boxscore for lineup: {e}")
            continue
        
        for side in ["home", "away"]:
            team_id = game.get(f"{side}_id", 0)
            batters_key = f"{side}Batters"
            batters = box.get(batters_key, [])
            
            for i, batter in enumerate(batters):
                if isinstance(batter, dict):
                    rows.append({
                        "game_pk": game_pk,
                        "game_date": date,
                        "team_id": team_id,
                        "batter_id": batter.get("personId", 0),
                        "batter_name": batter.get("name", ""),
                        "batting_order": batter.get("battingOrder", i + 1),
                        "position": batter.get("position", ""),
                        "handedness": batter.get("batSide", "R"),
                        "is_starter": i < 9,
                        "entered_as_sub": i >= 9,
                    })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    
    return df


def get_probable_starters(date: str) -> pd.DataFrame:
    """
    Get probable starting pitchers for games on a given date.
    Returns: game_pk, pitcher_id, pitcher_name, team_id, opponent_team_id,
             is_home, ballpark_id
    """
    games = load_schedule(date)
    rows = []
    
    for game in games:
        game_pk = game["game_id"]
        
        for side in ["home", "away"]:
            other_side = "away" if side == "home" else "home"
            pitcher_key = f"{side}_probable_pitcher"
            pitcher_name = game.get(pitcher_key, "")
            
            if pitcher_name:
                rows.append({
                    "game_pk": game_pk,
                    "game_date": date,
                    "pitcher_name": pitcher_name,
                    "team_id": game.get(f"{side}_id", 0),
                    "opponent_team_id": game.get(f"{other_side}_id", 0),
                    "is_home": side == "home",
                    "ballpark_id": str(game.get("venue_id", "")),
                })
    
    return pd.DataFrame(rows)
