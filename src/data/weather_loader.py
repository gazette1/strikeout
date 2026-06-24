"""
Weather data loader via Meteostat.
Provides game-time temperature and humidity for each ballpark.
"""
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_WEATHER, LEAGUE_AVG, PROJECT_ROOT


def _load_ballparks() -> dict:
    """Load ballpark config from YAML."""
    bp_path = PROJECT_ROOT / "config" / "ballparks.yaml"
    with open(bp_path) as f:
        data = yaml.safe_load(f)
    return {str(bp["venue_id"]): bp for bp in data.get("ballparks", data) if isinstance(bp, dict)}


def get_weather_for_game(ballpark_id: str, game_date: str,
                         game_hour_utc: int = 23) -> dict:
    """
    Get weather observation for a specific game.
    
    Args:
        ballpark_id: Venue ID string
        game_date: 'YYYY-MM-DD'
        game_hour_utc: Approximate game start hour in UTC (default 23 = ~7PM ET)
    
    Returns:
        dict with temperature_f, humidity_pct, wind_speed_mph, is_dome, roof_status
    """
    ballparks = _load_ballparks()
    bp = ballparks.get(ballpark_id, {})
    
    # Dome handling
    is_dome = bp.get("is_dome", False)
    roof_type = bp.get("roof_type", "open")
    
    if roof_type == "fixed":
        return {
            "temperature_f": LEAGUE_AVG["dome_temp_f"],
            "humidity_pct": LEAGUE_AVG["dome_humidity_pct"],
            "wind_speed_mph": 0.0,
            "is_dome": True,
            "roof_status": "closed",
        }
    
    lat = bp.get("latitude")
    lon = bp.get("longitude")
    elev = bp.get("elevation_m", 0)
    
    if lat is None or lon is None:
        logger.warning(f"No coordinates for ballpark {ballpark_id}, using defaults")
        return _default_weather(is_dome, roof_type)
    
    try:
        from meteostat import Point, Hourly
        
        location = Point(lat, lon, elev)
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        start = dt.replace(hour=max(0, game_hour_utc - 2))
        end = dt.replace(hour=min(23, game_hour_utc + 2))
        
        hourly = Hourly(location, start, end)
        data = hourly.fetch()
        
        if data.empty:
            logger.warning(f"No weather data for {ballpark_id} on {game_date}")
            return _default_weather(is_dome, roof_type)
        
        # Take the observation closest to game time
        row = data.iloc[len(data) // 2]
        
        temp_c = row.get("temp", np.nan)
        temp_f = temp_c * 9 / 5 + 32 if pd.notna(temp_c) else LEAGUE_AVG["dome_temp_f"]
        humidity = row.get("rhum", LEAGUE_AVG["dome_humidity_pct"])
        wspd_kmh = row.get("wspd", 0)
        wind_mph = wspd_kmh * 0.621371
        
        return {
            "temperature_f": round(temp_f, 1),
            "humidity_pct": round(humidity, 1) if pd.notna(humidity) else LEAGUE_AVG["dome_humidity_pct"],
            "wind_speed_mph": round(wind_mph, 1),
            "is_dome": is_dome,
            "roof_status": "retractable" if roof_type == "retractable" else "open",
        }
    
    except Exception as e:
        logger.warning(f"Weather fetch failed for {ballpark_id}: {e}")
        return _default_weather(is_dome, roof_type)


def _default_weather(is_dome: bool, roof_type: str) -> dict:
    """Return default weather values."""
    if is_dome or roof_type == "fixed":
        return {
            "temperature_f": LEAGUE_AVG["dome_temp_f"],
            "humidity_pct": LEAGUE_AVG["dome_humidity_pct"],
            "wind_speed_mph": 0.0,
            "is_dome": True,
            "roof_status": "closed",
        }
    return {
        "temperature_f": LEAGUE_AVG["dome_temp_f"],  # 72 as fallback
        "humidity_pct": LEAGUE_AVG["dome_humidity_pct"],
        "wind_speed_mph": 5.0,
        "is_dome": False,
        "roof_status": "open",
    }


def load_weather_for_date(date: str, games: pd.DataFrame) -> pd.DataFrame:
    """
    Load weather for all games on a date.
    
    Args:
        date: 'YYYY-MM-DD'
        games: DataFrame with game_pk and ballpark_id columns
    
    Returns:
        DataFrame with game_pk, temperature_f, humidity_pct, wind_speed_mph, is_dome, roof_status
    """
    cache_path = RAW_WEATHER / f"{date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    rows = []
    for _, game in games.iterrows():
        wx = get_weather_for_game(
            str(game.get("ballpark_id", "")),
            date,
        )
        wx["game_pk"] = game["game_pk"]
        wx["game_date"] = date
        wx["ballpark_id"] = str(game.get("ballpark_id", ""))
        rows.append(wx)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    
    return df
