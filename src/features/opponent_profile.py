"""
Opponent Profile features (19-31).
Derived from team batting data, lineups, and schedule context.
"""
import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    LEAGUE_AVG, LINEUP_SLOT_PA_WEIGHTS, SWING_DESCS,
    SWINGING_STRIKE_DESCS, CSW_DESCS, VELO_BANDS,
)

# Statcast zone IDs: 1-9 are in-zone; 11-14 are out-of-zone
IN_ZONE_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9}
OUT_ZONE_IDS = {11, 12, 13, 14}


def _get_team_pitches_vs_hand(pitches: pd.DataFrame, opp_team_id,
                               pitcher_hand: str, as_of_date: str,
                               current_season_only: bool = True) -> pd.DataFrame:
    """
    Filter pitches where batter_team == opp_team_id, faced a pitcher of
    the given hand, before as_of_date.
    Expects columns: batter_team (or away_team / home_team), p_throws, game_date.
    """
    df = pitches.copy()
    date_mask = pd.to_datetime(df["game_date"]) < pd.to_datetime(as_of_date)

    # Identify batter's team column (Statcast uses different conventions)
    team_col = None
    for col in ("batter_team", "batting_team", "inning_topbot"):
        if col in df.columns:
            team_col = col
            break

    if team_col == "inning_topbot":
        # Use inning_topbot + home_team/away_team to determine batter's team
        away_mask = df["inning_topbot"] == "Top"
        batter_team = np.where(away_mask, df.get("away_team", ""), df.get("home_team", ""))
        df["_batter_team"] = batter_team
        team_col = "_batter_team"
    elif team_col is None:
        # No team column — return empty
        return pd.DataFrame()

    hand_mask = df.get("p_throws", pd.Series("R", index=df.index)) == pitcher_hand
    team_mask = df[team_col] == opp_team_id

    mask = date_mask & hand_mask & team_mask
    result = df.loc[mask].copy()

    if current_season_only and not result.empty:
        year = pd.to_datetime(as_of_date).year
        result = result[pd.to_datetime(result["game_date"]).dt.year == year]

    return result


def _get_team_pitches(pitches: pd.DataFrame, opp_team_id,
                       as_of_date: str,
                       current_season_only: bool = True) -> pd.DataFrame:
    """Filter pitches where batter's team == opp_team_id before as_of_date."""
    return _get_team_pitches_vs_hand(pitches, opp_team_id, pitcher_hand=None,
                                      as_of_date=as_of_date,
                                      current_season_only=current_season_only)


def _get_team_pitches_any_hand(pitches: pd.DataFrame, opp_team_id,
                                as_of_date: str,
                                current_season_only: bool = True) -> pd.DataFrame:
    """Filter pitches for opp_team batters regardless of pitcher hand."""
    df = pitches.copy()
    date_mask = pd.to_datetime(df["game_date"]) < pd.to_datetime(as_of_date)

    team_col = None
    for col in ("batter_team", "batting_team"):
        if col in df.columns:
            team_col = col
            break

    if team_col is None and "inning_topbot" in df.columns:
        away_mask = df["inning_topbot"] == "Top"
        batter_team = np.where(away_mask,
                                df.get("away_team", pd.Series("", index=df.index)),
                                df.get("home_team", pd.Series("", index=df.index)))
        df["_batter_team"] = batter_team
        team_col = "_batter_team"
    elif team_col is None:
        return pd.DataFrame()

    team_mask = df[team_col] == opp_team_id
    mask = date_mask & team_mask
    result = df.loc[mask].copy()

    if current_season_only and not result.empty:
        year = pd.to_datetime(as_of_date).year
        result = result[pd.to_datetime(result["game_date"]).dt.year == year]

    return result


# ── Feature 19: Opponent team K% vs pitcher hand ──────────────
def feat_opp_team_k_rate_vs_hand(pitches: pd.DataFrame, opp_team_id,
                                   pitcher_hand: str, as_of_date: str) -> float:
    """
    K% for the opposing team when facing pitchers of the given hand.
    Blueprint Feature 19.
    """
    tp = _get_team_pitches_vs_hand(pitches, opp_team_id, pitcher_hand, as_of_date)

    if tp.empty:
        # Fall back: overall team K% regardless of hand
        tp_all = _get_team_pitches_any_hand(pitches, opp_team_id, as_of_date)
        if tp_all.empty:
            return LEAGUE_AVG["team_k_pct"]
        events = tp_all[tp_all["events"].notna()]
        if events.empty:
            return LEAGUE_AVG["team_k_pct"]
        k_count = (events["events"] == "strikeout").sum()
        return float(k_count / len(events))

    # Filter to plate-appearance-ending events
    events = tp[tp["events"].notna()]
    if events.empty:
        return LEAGUE_AVG["team_k_pct"]

    k_count = (events["events"] == "strikeout").sum()
    return float(k_count / len(events))


# ── Feature 20: Opponent O-Swing% ────────────────────────────
def feat_opp_o_swing_pct(pitches: pd.DataFrame, opp_team_id,
                          as_of_date: str) -> float:
    """
    Opponent team's chase rate (swing% on pitches outside zone).
    Blueprint Feature 20.
    """
    tp = _get_team_pitches_any_hand(pitches, opp_team_id, as_of_date)

    if tp.empty or "zone" not in tp.columns:
        return LEAGUE_AVG["o_swing_pct"]

    out_zone = tp[tp["zone"].isin(OUT_ZONE_IDS)]
    if out_zone.empty:
        return LEAGUE_AVG["o_swing_pct"]

    swings = out_zone["description"].isin(SWING_DESCS).sum()
    return float(swings / len(out_zone))


# ── Feature 21: Opponent Z-Contact% ──────────────────────────
def feat_opp_z_contact_pct(pitches: pd.DataFrame, opp_team_id,
                             as_of_date: str) -> float:
    """
    Opponent team's contact rate on in-zone pitches (zones 1-9).
    Blueprint Feature 21.
    """
    tp = _get_team_pitches_any_hand(pitches, opp_team_id, as_of_date)

    if tp.empty or "zone" not in tp.columns:
        return LEAGUE_AVG["z_contact_pct"]

    in_zone = tp[tp["zone"].isin(IN_ZONE_IDS)]
    if in_zone.empty:
        return LEAGUE_AVG["z_contact_pct"]

    # Swings in zone
    in_zone_swings = in_zone[in_zone["description"].isin(SWING_DESCS)]
    if in_zone_swings.empty:
        return LEAGUE_AVG["z_contact_pct"]

    # Contact = not a swinging strike
    contact = in_zone_swings[~in_zone_swings["description"].isin(SWINGING_STRIKE_DESCS)]
    return float(len(contact) / len(in_zone_swings))


# ── Feature 22: Opponent overall contact rate ─────────────────
def feat_opp_contact_rate(pitches: pd.DataFrame, opp_team_id,
                           as_of_date: str) -> float:
    """
    Opponent team's overall contact rate on swing attempts.
    Blueprint Feature 22.
    """
    tp = _get_team_pitches_any_hand(pitches, opp_team_id, as_of_date)

    if tp.empty:
        return LEAGUE_AVG["contact_pct"]

    swings = tp[tp["description"].isin(SWING_DESCS)]
    if swings.empty:
        return LEAGUE_AVG["contact_pct"]

    contact = swings[~swings["description"].isin(SWINGING_STRIKE_DESCS)]
    return float(len(contact) / len(swings))


# ── Feature 23: Projected lineup weighted K rate ──────────────
def feat_projected_lineup_weighted_k_rate(pitches: pd.DataFrame,
                                           lineup_df: pd.DataFrame,
                                           game_pk,
                                           opp_team_id,
                                           as_of_date: str) -> float:
    """
    Weighted average of individual batter K% using LINEUP_SLOT_PA_WEIGHTS.
    Blueprint Feature 23.
    """
    if lineup_df is None or lineup_df.empty:
        # Fall back to team K%
        return feat_opp_team_k_rate_vs_hand(pitches, opp_team_id, "R", as_of_date)

    # Filter lineup to this game and team
    game_lineup = lineup_df[
        (lineup_df.get("game_pk", pd.Series(dtype=int)) == game_pk) &
        (lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id)
    ] if "game_pk" in lineup_df.columns else lineup_df[
        lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id
    ]

    if game_lineup.empty:
        return feat_opp_team_k_rate_vs_hand(pitches, opp_team_id, "R", as_of_date)

    year = pd.to_datetime(as_of_date).year
    total_weight = 0.0
    weighted_k = 0.0

    for _, batter_row in game_lineup.iterrows():
        batter_id = batter_row.get("batter_id", batter_row.get("player_id", 0))
        slot = int(batter_row.get("batting_order", batter_row.get("lineup_slot", 5)))
        slot = max(1, min(9, slot))
        pa_weight = LINEUP_SLOT_PA_WEIGHTS.get(slot, 4.0)

        # Compute batter K% from pitches
        batter_mask = (
            (pitches.get("batter", pd.Series(dtype=int)) == batter_id) &
            (pd.to_datetime(pitches["game_date"]).dt.year == year) &
            (pd.to_datetime(pitches["game_date"]) < pd.to_datetime(as_of_date))
        )
        batter_pa = pitches.loc[batter_mask & pitches["events"].notna()]
        if len(batter_pa) >= 10:
            batter_k_pct = (batter_pa["events"] == "strikeout").sum() / len(batter_pa)
        else:
            batter_k_pct = LEAGUE_AVG["team_k_pct"]

        weighted_k += pa_weight * batter_k_pct
        total_weight += pa_weight

    if total_weight == 0:
        return LEAGUE_AVG["team_k_pct"]

    return float(weighted_k / total_weight)


# ── Feature 24: Opponent whiff vs pitcher's top pitch types ───
def feat_opp_whiff_vs_pitch_types(pitches: pd.DataFrame,
                                    opp_team_id,
                                    pitcher_id: int,
                                    as_of_date: str) -> float:
    """
    Opponent team's whiff rate against the pitcher's top 2 pitch types.
    Blueprint Feature 24.
    """
    # Get pitcher's top 2 pitch types this season
    year = pd.to_datetime(as_of_date).year
    pitcher_mask = (
        (pitches.get("pitcher", pd.Series(dtype=int)) == pitcher_id) &
        (pd.to_datetime(pitches["game_date"]).dt.year == year) &
        (pd.to_datetime(pitches["game_date"]) < pd.to_datetime(as_of_date))
    )
    pitcher_pitches = pitches.loc[pitcher_mask]

    if pitcher_pitches.empty or "pitch_type" not in pitcher_pitches.columns:
        return LEAGUE_AVG["swstr_pct"]

    top_types = pitcher_pitches["pitch_type"].value_counts().head(2).index.tolist()
    if not top_types:
        return LEAGUE_AVG["swstr_pct"]

    # Get opponent team pitches vs those pitch types
    tp = _get_team_pitches_any_hand(pitches, opp_team_id, as_of_date)
    if tp.empty or "pitch_type" not in tp.columns:
        return LEAGUE_AVG["swstr_pct"]

    relevant = tp[tp["pitch_type"].isin(top_types)]
    if relevant.empty:
        return LEAGUE_AVG["swstr_pct"]

    swings = relevant[relevant["description"].isin(SWING_DESCS)]
    if swings.empty:
        return LEAGUE_AVG["swstr_pct"]

    whiffs = swings[swings["description"].isin(SWINGING_STRIKE_DESCS)]
    return float(len(whiffs) / len(swings))


# ── Feature 25: Opponent chase rate vs velo band ──────────────
def feat_opp_chase_vs_velo_band(pitches: pd.DataFrame,
                                  opp_team_id,
                                  primary_velo: float,
                                  as_of_date: str) -> float:
    """
    Opponent O-Swing% on out-of-zone pitches within the pitcher's primary velo band.
    Blueprint Feature 25.
    """
    # Determine velo band
    band_range = VELO_BANDS["medium"]  # Default
    for band_name, (lo, hi) in VELO_BANDS.items():
        if lo <= primary_velo < hi:
            band_range = (lo, hi)
            break

    tp = _get_team_pitches_any_hand(pitches, opp_team_id, as_of_date)
    if tp.empty or "release_speed" not in tp.columns or "zone" not in tp.columns:
        return feat_opp_o_swing_pct(pitches, opp_team_id, as_of_date)

    velo_mask = (tp["release_speed"] >= band_range[0]) & (tp["release_speed"] < band_range[1])
    out_zone_mask = tp["zone"].isin(OUT_ZONE_IDS)
    relevant = tp[velo_mask & out_zone_mask]

    if relevant.empty:
        return feat_opp_o_swing_pct(pitches, opp_team_id, as_of_date)

    swings = relevant["description"].isin(SWING_DESCS).sum()
    return float(swings / len(relevant))


# ── Feature 26: Lineup handedness stack ───────────────────────
def feat_lineup_handedness_stack(lineup_df: pd.DataFrame,
                                  game_pk,
                                  opp_team_id,
                                  pitcher_hand: str) -> int:
    """
    Count of consecutive same-handedness batters that exploit the pitcher's
    weak split (lefties vs RHP, righties vs LHP).
    Returns ordinal 0-9; fallback is 4 (neutral).
    Blueprint Feature 26.
    """
    if lineup_df is None or lineup_df.empty:
        return 4  # Neutral fallback

    weak_hand = "L" if pitcher_hand == "R" else "R"

    # Filter lineup for this game/team
    if "game_pk" in lineup_df.columns:
        game_lineup = lineup_df[
            (lineup_df["game_pk"] == game_pk) &
            (lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id)
        ]
    else:
        game_lineup = lineup_df[
            lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id
        ]

    if game_lineup.empty:
        return 4

    # Sort by batting order
    slot_col = "batting_order" if "batting_order" in game_lineup.columns else "lineup_slot"
    if slot_col not in game_lineup.columns:
        return 4

    game_lineup = game_lineup.sort_values(slot_col)

    hand_col = "batter_hand" if "batter_hand" in game_lineup.columns else "bat_side"
    if hand_col not in game_lineup.columns:
        return 4

    # Find longest consecutive run of weak_hand batters
    max_run = 0
    current_run = 0
    for _, row in game_lineup.iterrows():
        if row[hand_col] == weak_hand:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return int(min(max_run, 9))


# ── Feature 27: Opponent plate discipline variance ────────────
def feat_opp_plate_discipline_variance(pitches: pd.DataFrame,
                                        lineup_df: pd.DataFrame,
                                        game_pk,
                                        opp_team_id,
                                        as_of_date: str) -> float:
    """
    Std dev of individual batter K% in the projected lineup.
    High variance = exploitable weak spots. Blueprint Feature 27.
    """
    if lineup_df is None or lineup_df.empty:
        return 0.05  # League median approximation

    if "game_pk" in lineup_df.columns:
        game_lineup = lineup_df[
            (lineup_df["game_pk"] == game_pk) &
            (lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id)
        ]
    else:
        game_lineup = lineup_df[
            lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id
        ]

    if game_lineup.empty:
        return 0.05

    year = pd.to_datetime(as_of_date).year
    batter_k_pcts = []

    batter_id_col = "batter_id" if "batter_id" in game_lineup.columns else "player_id"
    if batter_id_col not in game_lineup.columns:
        return 0.05

    for _, batter_row in game_lineup.iterrows():
        batter_id = batter_row[batter_id_col]
        batter_mask = (
            (pitches.get("batter", pd.Series(dtype=int)) == batter_id) &
            (pd.to_datetime(pitches["game_date"]).dt.year == year) &
            (pd.to_datetime(pitches["game_date"]) < pd.to_datetime(as_of_date))
        )
        batter_pa = pitches.loc[batter_mask & pitches["events"].notna()]
        if len(batter_pa) >= 20:
            batter_k_pct = (batter_pa["events"] == "strikeout").sum() / len(batter_pa)
            batter_k_pcts.append(batter_k_pct)

    if len(batter_k_pcts) < 2:
        return 0.05

    return float(np.std(batter_k_pcts))


# ── Feature 28: Lineup sub risk ───────────────────────────────
def feat_lineup_sub_risk(lineup_df: pd.DataFrame,
                          game_pk,
                          opp_team_id,
                          as_of_date: str) -> float:
    """
    Fraction of projected lineup batters with <50% start rate in last 14 days.
    Blueprint Feature 28.
    """
    if lineup_df is None or lineup_df.empty:
        return LEAGUE_AVG["lineup_sub_risk"]

    if "game_pk" in lineup_df.columns:
        game_lineup = lineup_df[
            (lineup_df["game_pk"] == game_pk) &
            (lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id)
        ]
    else:
        game_lineup = lineup_df[
            lineup_df.get("team_id", pd.Series(dtype=int)) == opp_team_id
        ]

    if game_lineup.empty:
        return LEAGUE_AVG["lineup_sub_risk"]

    # Look for start_rate or games_started_last_14 column
    if "start_rate_14d" in game_lineup.columns:
        sub_risk = (game_lineup["start_rate_14d"] < 0.50).mean()
        return float(sub_risk)
    elif "games_started_14d" in game_lineup.columns:
        sub_risk = (game_lineup["games_started_14d"] < 7).mean()
        return float(sub_risk)
    else:
        # No start rate data available
        return LEAGUE_AVG["lineup_sub_risk"]


# ── Feature 29: Matchup familiarity ───────────────────────────
def feat_matchup_familiarity(game_logs: pd.DataFrame,
                               pitcher_id: int,
                               opp_team_id,
                               as_of_date: str) -> int:
    """
    Number of times pitcher has faced opp_team_id this season (capped at 3).
    Blueprint Feature 29.
    """
    year = pd.to_datetime(as_of_date).year

    opp_col = None
    for col in ("opponent_team_id", "opp_team_id", "opponent_id", "away_team", "home_team"):
        if col in game_logs.columns:
            opp_col = col
            break

    if opp_col is None:
        return 0

    mask = (
        (game_logs["pitcher_id"] == pitcher_id) &
        (pd.to_datetime(game_logs["game_date"]).dt.year == year) &
        (pd.to_datetime(game_logs["game_date"]) < pd.to_datetime(as_of_date)) &
        (game_logs[opp_col] == opp_team_id)
    )

    count = mask.sum()
    return int(min(count, 3))


# ── Feature 30: Opponent travel fatigue ───────────────────────
def feat_opp_travel_fatigue(schedule_data,
                              opp_team_id,
                              game_date: str) -> int:
    """
    Ordinal 0-2 encoding opponent travel fatigue:
      0 = home or no recent travel
      1 = moderate travel (one time zone)
      2 = heavy travel (cross-country or back-to-back road)
    Blueprint Feature 30.
    """
    if schedule_data is None:
        return 0

    if isinstance(schedule_data, pd.DataFrame):
        if schedule_data.empty:
            return 0

        # Look for recent games for this team before game_date
        try:
            team_games = schedule_data[
                (schedule_data.get("away_team", pd.Series(dtype=str)) == opp_team_id) |
                (schedule_data.get("home_team", pd.Series(dtype=str)) == opp_team_id)
            ]
            game_dt = pd.to_datetime(game_date)
            recent = team_games[
                (pd.to_datetime(team_games["game_date"]) < game_dt) &
                (pd.to_datetime(team_games["game_date"]) >= game_dt - pd.Timedelta(days=3))
            ].sort_values("game_date", ascending=False)

            if recent.empty:
                return 0

            # Check if the team was on the road for consecutive games
            is_home_col = "home_team"
            if is_home_col in recent.columns:
                road_games = (recent[is_home_col] != opp_team_id).sum()
                if road_games >= 2:
                    return 2
                elif road_games == 1:
                    return 1
            return 0
        except Exception:
            return 0
    return 0


# ── Feature 31: Opponent game importance ─────────────────────
def feat_opp_game_importance(game_date: str, standings_gb: float = 5.0) -> int:
    """
    Encode game importance based on calendar month and standings.
      0 = low (April/early May or far from contention)
      1 = medium (May-July)
      2 = high (Aug/Sept or within 3 GB of wild card)
    Blueprint Feature 31.
    """
    month = pd.to_datetime(game_date).month

    if standings_gb is None:
        standings_gb = 5.0

    # High importance: September/October pennant race or within striking distance
    if month >= 9 or standings_gb <= 3.0:
        return 2
    # Medium importance: late spring to midsummer
    elif month >= 5 or standings_gb <= 7.0:
        return 1
    # Low importance: early season
    else:
        return 0


def compute_all_opponent_profile(pitches: pd.DataFrame,
                                  game_logs: pd.DataFrame,
                                  lineup_df: pd.DataFrame,
                                  pitcher_id: int,
                                  opp_team_id,
                                  as_of_date: str,
                                  game_pk,
                                  pitcher_hand: str = "R",
                                  primary_velo: float = 93.0,
                                  standings_gb: float = 5.0,
                                  schedule_data=None) -> dict:
    """Compute all 13 opponent profile features. Returns dict of feature_name -> value."""
    return {
        "opp_team_k_rate_vs_hand": feat_opp_team_k_rate_vs_hand(
            pitches, opp_team_id, pitcher_hand, as_of_date),
        "opp_o_swing_pct": feat_opp_o_swing_pct(
            pitches, opp_team_id, as_of_date),
        "opp_z_contact_pct": feat_opp_z_contact_pct(
            pitches, opp_team_id, as_of_date),
        "opp_contact_rate": feat_opp_contact_rate(
            pitches, opp_team_id, as_of_date),
        "projected_lineup_weighted_k_rate": feat_projected_lineup_weighted_k_rate(
            pitches, lineup_df, game_pk, opp_team_id, as_of_date),
        "opp_whiff_vs_pitch_types": feat_opp_whiff_vs_pitch_types(
            pitches, opp_team_id, pitcher_id, as_of_date),
        "opp_chase_vs_velo_band": feat_opp_chase_vs_velo_band(
            pitches, opp_team_id, primary_velo, as_of_date),
        "lineup_handedness_stack": feat_lineup_handedness_stack(
            lineup_df, game_pk, opp_team_id, pitcher_hand),
        "opp_plate_discipline_variance": feat_opp_plate_discipline_variance(
            pitches, lineup_df, game_pk, opp_team_id, as_of_date),
        "lineup_sub_risk": feat_lineup_sub_risk(
            lineup_df, game_pk, opp_team_id, as_of_date),
        "matchup_familiarity": feat_matchup_familiarity(
            game_logs, pitcher_id, opp_team_id, as_of_date),
        "opp_travel_fatigue": feat_opp_travel_fatigue(
            schedule_data, opp_team_id, as_of_date),
        "opp_game_importance": feat_opp_game_importance(
            as_of_date, standings_gb),
    }
