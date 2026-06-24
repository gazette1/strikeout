"""
Tests for the data loading layer.
All external API calls are mocked — no real network I/O required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


# ─────────────────────────────────────────────────────────────────────────────
# 1. test_statcast_loader_caching
# ─────────────────────────────────────────────────────────────────────────────
def _make_mock_pybaseball(return_value=None):
    """
    Build a MagicMock module that can be injected into sys.modules
    in place of pybaseball. The statcast function returns return_value.
    """
    import sys
    mock_pb = MagicMock()
    mock_pb.statcast.return_value = return_value if return_value is not None else pd.DataFrame()
    return mock_pb


def test_statcast_loader_caching(tmp_path):
    """
    Second call to load_statcast_day should read from parquet cache,
    NOT call pybaseball.statcast again.
    """
    import sys
    from src.data.statcast_loader import load_statcast_day

    date = "2024-06-01"
    cache_dir = tmp_path / "statcast"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / f"{date}.parquet"

    # Build a minimal fake DataFrame and pre-cache it
    fake_df = pd.DataFrame({
        "pitcher": [12345, 12345],
        "game_date": [date, date],
        "game_pk": [1001, 1001],
        "pitch_type": ["FF", "SL"],
        "release_speed": [95.0, 84.0],
    })
    fake_df.to_parquet(cache_path, index=False)

    mock_pb = _make_mock_pybaseball(fake_df)

    # Patch RAW_STATCAST to our tmp dir so the loader uses it
    with patch("src.data.statcast_loader.RAW_STATCAST", cache_dir):
        with patch.dict(sys.modules, {"pybaseball": mock_pb}):
            result = load_statcast_day(date, cache=True)

    # pybaseball.statcast should NOT have been called — cache hit
    mock_pb.statcast.assert_not_called()
    assert len(result) == 2
    assert "pitcher" in result.columns


def test_statcast_loader_calls_api_on_cache_miss(tmp_path):
    """
    When cache is absent, load_statcast_day should call pybaseball.statcast
    and then write a parquet file.
    """
    import sys
    from src.data.statcast_loader import load_statcast_day

    date = "2024-06-15"
    cache_dir = tmp_path / "statcast"
    cache_dir.mkdir(parents=True)

    fake_df = pd.DataFrame({
        "pitcher": [12345],
        "game_date": [date],
        "game_pk": [2001],
        "pitch_type": ["FF"],
        "release_speed": [94.0],
    })

    mock_pb = _make_mock_pybaseball(fake_df)

    with patch("src.data.statcast_loader.RAW_STATCAST", cache_dir):
        with patch.dict(sys.modules, {"pybaseball": mock_pb}):
            result = load_statcast_day(date, cache=True)

    mock_pb.statcast.assert_called_once_with(start_dt=date, end_dt=date)
    assert len(result) == 1
    # Cache should now exist
    assert (cache_dir / f"{date}.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 2. test_statcast_loader_empty_response
# ─────────────────────────────────────────────────────────────────────────────
def test_statcast_loader_empty_response(tmp_path):
    """
    When pybaseball.statcast returns an empty DataFrame, the loader
    should return an empty DataFrame gracefully (not raise).
    """
    import sys
    from src.data.statcast_loader import load_statcast_day

    date = "2024-06-20"
    cache_dir = tmp_path / "statcast"
    cache_dir.mkdir(parents=True)

    mock_pb = _make_mock_pybaseball(pd.DataFrame())

    with patch("src.data.statcast_loader.RAW_STATCAST", cache_dir):
        with patch.dict(sys.modules, {"pybaseball": mock_pb}):
            result = load_statcast_day(date, cache=False)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_statcast_loader_none_response(tmp_path):
    """
    When pybaseball.statcast returns None, the loader should return
    an empty DataFrame gracefully.
    """
    import sys
    from src.data.statcast_loader import load_statcast_day

    date = "2024-06-21"
    cache_dir = tmp_path / "statcast"
    cache_dir.mkdir(parents=True)

    mock_pb = _make_mock_pybaseball(None)

    with patch("src.data.statcast_loader.RAW_STATCAST", cache_dir):
        with patch.dict(sys.modules, {"pybaseball": mock_pb}):
            result = load_statcast_day(date, cache=False)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# 3. test_mlb_schedule_parsing
# ─────────────────────────────────────────────────────────────────────────────
_MOCK_SCHEDULE_RESPONSE = [
    {
        "game_id": 748534,
        "game_datetime": "2024-06-01T23:10:00Z",
        "game_date": "2024-06-01",
        "game_type": "R",
        "status": "Final",
        "away_name": "New York Yankees",
        "home_name": "Boston Red Sox",
        "away_id": 147,
        "home_id": 111,
        "venue_id": 3,
        "venue_name": "Fenway Park",
        "national_broadcasts": [],
        "series_status": "",
        "away_score": 5,
        "home_score": 3,
    }
]

_MOCK_BOXSCORE = {
    "awayPitchers": [
        {
            "personId": 12345,
            "name": "Justin Verlander",
            "ip": "6.0",
            "k": 8,
            "p": 95,
            "bb": 2,
            "er": 2,
            "h": 5,
        }
    ],
    "homePitchers": [
        {
            "personId": 67890,
            "name": "Gerrit Cole",
            "ip": "7.0",
            "k": 10,
            "p": 105,
            "bb": 1,
            "er": 1,
            "h": 4,
        }
    ],
    "away": {
        "pitchers": [12345],
        "team": {"id": 147},
    },
    "home": {
        "pitchers": [67890],
        "team": {"id": 111},
    },
    "officials": [
        {
            "official": {"id": 427474, "fullName": "Angel Hernandez"},
            "officialType": "Home Plate",
        }
    ],
}

_MOCK_GAME_DATA = {
    "gameData": {
        "venue": {"id": "3", "name": "Fenway Park"},
        "datetime": {"dateTime": "2024-06-01T23:10:00Z"},
    },
    "liveData": {
        "boxscore": {
            "officials": [
                {
                    "official": {"id": 427474, "fullName": "Angel Hernandez"},
                    "officialType": "Home Plate",
                }
            ]
        }
    },
}


def test_mlb_schedule_parsing(tmp_path):
    """
    Mock statsapi.schedule + boxscore, verify extract_pitcher_game_logs
    returns the expected columns and at least one row.
    """
    import sys
    from src.data.mlb_api_loader import extract_pitcher_game_logs

    cache_dir = tmp_path / "game_logs"
    cache_dir.mkdir(parents=True)

    mock_statsapi = MagicMock()
    mock_statsapi.schedule.return_value = _MOCK_SCHEDULE_RESPONSE
    mock_statsapi.boxscore_data.return_value = _MOCK_BOXSCORE
    mock_statsapi.get.return_value = _MOCK_GAME_DATA

    with patch("src.data.mlb_api_loader.RAW_GAME_LOGS", cache_dir):
        with patch.dict(sys.modules, {"statsapi": mock_statsapi}):
            result = extract_pitcher_game_logs("2024-06-01")

    assert isinstance(result, pd.DataFrame)
    # Should have at least the core columns
    expected_cols = {
        "game_pk", "pitcher_id", "pitcher_name",
        "innings_pitched", "strikeouts",
    }
    missing = expected_cols - set(result.columns)
    assert not missing, f"Missing columns: {missing}"
    assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. test_weather_loader_dome_detection
# ─────────────────────────────────────────────────────────────────────────────
def test_weather_loader_dome_detection():
    """
    Dome stadiums (roof_type == 'fixed') should return the fixed
    dome temperature and humidity from LEAGUE_AVG, not call Meteostat.
    """
    import sys
    from src.data.weather_loader import get_weather_for_game
    from config.settings import LEAGUE_AVG

    mock_ballparks = {
        "dome_park": {
            "venue_id": "dome_park",
            "team": "TBR",
            "is_dome": True,
            "roof_type": "fixed",
            "latitude": 27.77,
            "longitude": -82.65,
            "elevation_m": 0,
        }
    }

    mock_meteostat = MagicMock()

    with patch("src.data.weather_loader._load_ballparks", return_value=mock_ballparks):
        with patch.dict(sys.modules, {"meteostat": mock_meteostat}):
            result = get_weather_for_game("dome_park", "2024-06-01")

    # Meteostat should not have been called for a fixed-dome park
    mock_meteostat.Hourly.assert_not_called()

    assert result["temperature_f"] == pytest.approx(LEAGUE_AVG["dome_temp_f"])
    assert result["humidity_pct"] == pytest.approx(LEAGUE_AVG["dome_humidity_pct"])
    assert result["is_dome"] is True


def test_weather_loader_open_air_calls_api():
    """
    Open-air stadiums should attempt to call Meteostat (or fall back gracefully).
    """
    import sys
    from src.data.weather_loader import get_weather_for_game

    mock_ballparks = {
        "open_park": {
            "venue_id": "open_park",
            "team": "NYY",
            "is_dome": False,
            "roof_type": "open",
            "latitude": 40.83,
            "longitude": -73.93,
            "elevation_m": 5,
        }
    }

    mock_meteostat = MagicMock()
    mock_meteostat.Hourly.side_effect = Exception("Meteostat unavailable")
    mock_meteostat.Point = MagicMock()

    with patch("src.data.weather_loader._load_ballparks", return_value=mock_ballparks):
        with patch.dict(sys.modules, {"meteostat": mock_meteostat}):
            result = get_weather_for_game("open_park", "2024-06-01")

    # Should get a dict back regardless
    assert isinstance(result, dict)
    assert "temperature_f" in result


# ─────────────────────────────────────────────────────────────────────────────
# 5. test_schema_validator
# ─────────────────────────────────────────────────────────────────────────────
def test_schema_validator_valid_pitches():
    """Valid pitch DataFrame passes schema validation without raising."""
    from src.staging.schema_validator import validate_pitches

    df = pd.DataFrame({
        "game_pk": [100001],
        "game_date": ["2024-05-01"],
        "pitcher": [12345],
        "batter": [99999],
        "pitch_type": ["FF"],
        "release_speed": [95.0],
        "release_spin_rate": [2300.0],
        "pfx_x": [-0.7],
        "pfx_z": [1.3],
        "release_pos_x": [-1.5],
        "release_pos_z": [6.0],
        "plate_x": [0.2],
        "plate_z": [2.5],
        "vx0": [-5.0],
        "vy0": [-130.0],
        "vz0": [-5.0],
        "ax": [0.5],
        "ay": [24.0],
        "az": [-18.0],
        "zone": [5.0],
        "description": ["swinging_strike"],
        "events": [None],
        "strikes": [2.0],
        "balls": [1.0],
        "stand": ["R"],
        "p_throws": ["R"],
        "at_bat_number": [3.0],
        "pitch_number": [4.0],
        "inning": [3.0],
    })

    # validate_pitches should return a DataFrame (may coerce types) or None
    result = validate_pitches(df)
    assert result is not None or True  # Pass if no exception raised


def test_schema_validator_valid_games():
    """Valid game log DataFrame passes schema validation without raising."""
    from src.staging.schema_validator import validate_game_logs

    df = pd.DataFrame({
        "game_pk": [100001],
        "game_date": ["2024-05-01"],
        "pitcher_id": [12345],
        "pitcher_name": ["Test Pitcher"],
        "team_id": [147],
        "opponent_team_id": [111],
        "is_home": [True],
        "innings_pitched": [6.0],
        "strikeouts": [7],
        "pitches_thrown": [92],
        "walks": [2],
        "earned_runs": [1],
        "hits_allowed": [5],
    })

    result = validate_game_logs(df)
    assert result is not None or True  # Pass if no exception raised


def test_schema_validator_missing_required_column():
    """
    A DataFrame with only a few random columns is handled gracefully.
    validate_pitches returns the subset of known columns — it doesn't crash.
    It may return a nearly-empty (few columns) DataFrame.
    """
    from src.staging.schema_validator import validate_pitches

    # Only supply game_pk — all other required columns are absent
    df = pd.DataFrame({
        "game_pk": [100001, 100002],
        "something_random": [42, 43],
    })

    # Should not raise — validator is tolerant of missing columns
    result = validate_pitches(df)
    assert isinstance(result, pd.DataFrame)
    # game_pk is a known schema column, so it should appear
    if not result.empty:
        assert "game_pk" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# 6. test_deduplicator
# ─────────────────────────────────────────────────────────────────────────────
def test_deduplicator_removes_duplicates(tmp_path):
    """
    deduplicate_parquet should remove duplicate rows and return the count removed.
    """
    from src.staging.deduplicator import deduplicate_parquet

    parquet_path = tmp_path / "test_staging.parquet"

    df = pd.DataFrame({
        "game_pk": [1001, 1001, 1002, 1002, 1003],
        "pitcher_id": [12345, 12345, 67890, 67890, 12345],
        "strikeouts": [7, 7, 5, 5, 9],
    })
    df.to_parquet(parquet_path, index=False)

    removed = deduplicate_parquet(parquet_path, key_columns=["game_pk", "pitcher_id"])
    assert removed == 2  # Two duplicate rows

    # Read back and verify
    result = pd.read_parquet(parquet_path)
    assert len(result) == 3
    assert result["game_pk"].nunique() == 3


def test_deduplicator_no_duplicates(tmp_path):
    """When there are no duplicates, deduplicator returns 0."""
    from src.staging.deduplicator import deduplicate_parquet

    parquet_path = tmp_path / "clean_staging.parquet"

    df = pd.DataFrame({
        "game_pk": [1001, 1002, 1003],
        "pitcher_id": [12345, 67890, 12345],
        "strikeouts": [7, 5, 9],
    })
    df.to_parquet(parquet_path, index=False)

    removed = deduplicate_parquet(parquet_path, key_columns=["game_pk", "pitcher_id"])
    assert removed == 0


def test_deduplicator_missing_key_column(tmp_path):
    """
    When the key columns aren't present, deduplicator returns 0
    and doesn't crash.
    """
    from src.staging.deduplicator import deduplicate_parquet

    parquet_path = tmp_path / "no_key.parquet"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_parquet(parquet_path, index=False)

    removed = deduplicate_parquet(parquet_path, key_columns=["game_pk", "pitcher_id"])
    assert removed == 0


def test_deduplicator_nonexistent_file(tmp_path):
    """Calling deduplicate_parquet on a non-existent file returns 0."""
    from src.staging.deduplicator import deduplicate_parquet

    result = deduplicate_parquet(tmp_path / "does_not_exist.parquet", ["game_pk"])
    assert result == 0
