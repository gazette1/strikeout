"""
lineup_resolver.py
------------------
Resolves projected lineups and umpire assignments for a prediction date.

Functions
---------
get_probable_starters(date) -> pd.DataFrame
    Retrieve probable starting pitchers for all games on *date* and cache
    results to data/raw/lineups/{date}_starters.parquet.

resolve_lineups(date, game_pk) -> pd.DataFrame
    Attempt to pull the confirmed batting order for a game; return an
    empty DataFrame when the lineup is not yet posted.

get_umpire_assignment(date, game_pk) -> int
    Extract the home-plate umpire ID from live game data; return 0 when
    the assignment is unavailable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
from typing import Optional

import pandas as pd
import statsapi
from loguru import logger

from config.settings import RAW_LINEUPS, MLB_API_DELAY_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _throttle() -> None:
    """Pause execution to respect the MLB Stats API rate limit."""
    time.sleep(MLB_API_DELAY_SECONDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_probable_starters(date: str) -> pd.DataFrame:
    """
    Retrieve probable starting pitchers for all games on *date*.

    Calls ``statsapi.schedule`` to get the day's games, then extracts the
    home and away probable pitchers from the ``home_probable_pitcher`` /
    ``away_probable_pitcher`` fields.  Results are cached to
    ``data/raw/lineups/{date}_starters.parquet``.

    Parameters
    ----------
    date : str
        Date string in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        One row per probable starter with columns:
        ``pitcher_id``, ``pitcher_name``, ``team_id``,
        ``opponent_team_id``, ``game_pk``, ``ballpark_id``,
        ``pitcher_hand``, ``is_home``.
        Returns an empty DataFrame when no games are found.
    """
    cache_path = RAW_LINEUPS / f"{date}_starters.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached probable starters for {date}")
        return pd.read_parquet(cache_path)

    logger.info(f"Fetching probable starters for {date} from MLB Stats API")

    try:
        games: list[dict] = statsapi.schedule(start_date=date, end_date=date)
    except Exception as exc:
        logger.error(f"statsapi.schedule failed for {date}: {exc}")
        return pd.DataFrame()

    _throttle()

    rows: list[dict] = []

    for game in games:
        game_pk: int = game.get("game_id", 0)
        venue_id: str = str(game.get("venue_id", ""))

        # Attempt to pull additional detail (pitcher hand, pitcher ID) from
        # the live game feed.  The statsapi.schedule dict only carries the
        # pitcher name, so we need the game endpoint for the numeric ID and
        # handedness.
        try:
            game_detail: dict = statsapi.get("game", {"gamePk": game_pk})
            _throttle()
        except Exception as exc:
            logger.warning(f"Could not load game detail for game_pk={game_pk}: {exc}")
            game_detail = {}

        game_data: dict = game_detail.get("gameData", {})
        probable_pitchers: dict = game_data.get("probablePitchers", {})
        players: dict = game_data.get("players", {})

        teams: dict = game_data.get("teams", {})
        home_team_id: int = teams.get("home", {}).get("id", game.get("home_id", 0))
        away_team_id: int = teams.get("away", {}).get("id", game.get("away_id", 0))

        for side, team_id, opp_team_id in [
            ("home", home_team_id, away_team_id),
            ("away", away_team_id, home_team_id),
        ]:
            # Pitcher name from the simple schedule dict (fallback)
            pitcher_name_fallback: str = game.get(f"{side}_probable_pitcher", "")

            pitcher_detail: dict = probable_pitchers.get(side, {})
            pitcher_id: int = pitcher_detail.get("id", 0)
            pitcher_name: str = pitcher_detail.get("fullName", pitcher_name_fallback)

            if not pitcher_name:
                # No probable pitcher announced for this side — skip
                continue

            # Attempt to resolve handedness from the players dict
            pitcher_hand: str = "R"  # sane default
            if pitcher_id:
                player_key = f"ID{pitcher_id}"
                p_info: dict = players.get(player_key, {})
                pitcher_hand = (
                    p_info.get("pitchHand", {}).get("code", "R") or "R"
                )

            rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": date,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": pitcher_name,
                    "team_id": team_id,
                    "opponent_team_id": opp_team_id,
                    "is_home": side == "home",
                    "ballpark_id": venue_id,
                    "pitcher_hand": pitcher_hand,
                }
            )

    df = pd.DataFrame(rows)

    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(df)} probable starters for {date} → {cache_path}")
    else:
        logger.warning(f"No probable starters found for {date}")

    return df


def resolve_lineups(date: str, game_pk: int) -> pd.DataFrame:
    """
    Attempt to pull the confirmed batting order for a specific game.

    Uses ``statsapi.get("game", ...)`` to extract the official batting
    order when it has been posted.

    Parameters
    ----------
    date : str
        Date string ``YYYY-MM-DD`` (used for cache key only).
    game_pk : int
        The MLB game primary key.

    Returns
    -------
    pd.DataFrame
        Columns: ``batter_id``, ``batting_order``, ``batter_hand``,
        ``team_id``.  Returns an **empty** DataFrame when the lineup
        is not yet available or an error occurs.
    """
    logger.info(f"Resolving batting lineup for game_pk={game_pk} ({date})")

    try:
        game_data: dict = statsapi.get("game", {"gamePk": game_pk})
        _throttle()
    except Exception as exc:
        logger.warning(f"Failed to fetch game data for game_pk={game_pk}: {exc}")
        return pd.DataFrame()

    live_data: dict = game_data.get("liveData", {})
    boxscore: dict = live_data.get("boxscore", {})
    teams: dict = boxscore.get("teams", {})

    # Also gather player info for handedness
    players: dict = game_data.get("gameData", {}).get("players", {})

    rows: list[dict] = []

    for side in ("home", "away"):
        side_data: dict = teams.get(side, {})
        team_id: int = (
            side_data.get("team", {}).get("id", 0)
            or game_data.get("gameData", {})
            .get("teams", {})
            .get(side, {})
            .get("id", 0)
        )

        batters: dict = side_data.get("batters", [])
        batting_order_raw: list = side_data.get("battingOrder", [])
        all_players: dict = side_data.get("players", {})

        # battingOrder is a list of player IDs in order
        if batting_order_raw:
            for order_idx, player_id in enumerate(batting_order_raw, start=1):
                player_key = f"ID{player_id}"
                p_info: dict = all_players.get(player_key, {})
                person: dict = p_info.get("person", {})

                # Handedness from gameData players
                gd_player: dict = players.get(player_key, {})
                bat_side: str = (
                    gd_player.get("batSide", {}).get("code", "R") or "R"
                )

                rows.append(
                    {
                        "batter_id": player_id,
                        "batting_order": order_idx,
                        "batter_hand": bat_side,
                        "team_id": team_id,
                    }
                )
        elif batters:
            # Fallback: iterate the batters list directly
            for order_idx, player_id in enumerate(batters, start=1):
                player_key = f"ID{player_id}"
                gd_player: dict = players.get(player_key, {})
                bat_side: str = (
                    gd_player.get("batSide", {}).get("code", "R") or "R"
                )
                rows.append(
                    {
                        "batter_id": player_id,
                        "batting_order": order_idx,
                        "batter_hand": bat_side,
                        "team_id": team_id,
                    }
                )

    if not rows:
        logger.debug(f"No batting order available for game_pk={game_pk}")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def get_umpire_assignment(date: str, game_pk: int) -> int:
    """
    Extract the home-plate umpire ID for a game.

    Parameters
    ----------
    date : str
        Date string ``YYYY-MM-DD`` (used for logging only).
    game_pk : int
        The MLB game primary key.

    Returns
    -------
    int
        Numeric umpire ID, or ``0`` if the assignment cannot be found.
    """
    logger.info(f"Fetching umpire assignment for game_pk={game_pk} ({date})")

    try:
        game_data: dict = statsapi.get("game", {"gamePk": game_pk})
        _throttle()
    except Exception as exc:
        logger.warning(f"Failed to fetch game data for umpire lookup game_pk={game_pk}: {exc}")
        return 0

    try:
        officials: list = (
            game_data.get("liveData", {})
            .get("boxscore", {})
            .get("officials", [])
        )
        for official in officials:
            if official.get("officialType") == "Home Plate":
                umpire_id: int = official.get("official", {}).get("id", 0)
                logger.debug(
                    f"Home plate umpire ID={umpire_id} for game_pk={game_pk}"
                )
                return int(umpire_id)
    except Exception as exc:
        logger.warning(f"Error parsing umpire data for game_pk={game_pk}: {exc}")

    return 0
