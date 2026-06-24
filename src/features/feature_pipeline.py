"""
Feature pipeline orchestrator.
Computes all 40 features for probable starters on a given date.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import STAGING_PITCHES, STAGING_GAMES, FEATURES_DIR
from src.features.pitcher_ability import compute_all_pitcher_ability
from src.features.recent_form import compute_all_recent_form
from src.features.opponent_profile import compute_all_opponent_profile
from src.features.contextual import compute_all_contextual
from src.features.battery_effects import compute_all_battery_effects
from src.features.imputation import impute_features, impute_dataframe, get_feature_fill_report

# Ordered feature columns matching features.yaml
FEATURE_COLUMNS = [
    "k_per_9_rolling", "swstr_pct", "csw_pct", "pitch_mix_k_profile",
    "fb_velo_avg", "fb_velo_max", "fb_spin_rate", "vaa_fastball",
    "ivb_delta", "hb_delta", "release_point_consistency", "arm_slot_drift",
    "tunneling_efficiency", "two_strike_putaway_rate",
    "k_rate_last_5", "recent_pitch_count_trend", "ip_per_start", "days_rest",
    "opp_team_k_rate_vs_hand", "opp_o_swing_pct", "opp_z_contact_pct",
    "opp_contact_rate", "projected_lineup_weighted_k_rate",
    "opp_whiff_vs_pitch_types", "opp_chase_vs_velo_band",
    "lineup_handedness_stack", "opp_plate_discipline_variance",
    "lineup_sub_risk", "matchup_familiarity", "opp_travel_fatigue",
    "opp_game_importance",
    "park_k_factor", "ump_k_boost", "temperature_f", "humidity_pct",
    "pitcher_leash",
    "catcher_framing_runs", "battery_k_rate_together",
    "battery_k_rate_delta", "catcher_game_calling_aggression",
]

METADATA_COLUMNS = ["pitcher_id", "game_date", "game_pk", "opponent_team_id"]


def load_staging_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pitches and game logs from staging parquet files."""
    pitches = pd.DataFrame()
    games = pd.DataFrame()

    if STAGING_PITCHES.exists():
        try:
            pitches = pd.read_parquet(STAGING_PITCHES)
            logger.info(f"Loaded {len(pitches):,} pitch rows from staging")
        except Exception as exc:
            logger.error(f"Failed to load staging pitches: {exc}")

    if STAGING_GAMES.exists():
        try:
            games = pd.read_parquet(STAGING_GAMES)
            logger.info(f"Loaded {len(games):,} game rows from staging")
        except Exception as exc:
            logger.error(f"Failed to load staging games: {exc}")

    return pitches, games


def _load_umpire_history():
    """
    Load umpire history from the data layer.
    Returns an empty DataFrame if the loader is not available.
    """
    try:
        from src.data.umpire_loader import load_umpire_history
        return load_umpire_history()
    except ImportError:
        logger.debug("umpire_loader not available; ump_k_boost will be NaN")
        return pd.DataFrame()
    except Exception as exc:
        logger.warning(f"Could not load umpire history: {exc}")
        return pd.DataFrame()


def compute_features_for_pitcher(
    pitches: pd.DataFrame,
    game_logs: pd.DataFrame,
    lineups: pd.DataFrame,
    weather_df: pd.DataFrame,
    umpire_history,
    pitcher_id: int,
    opponent_team_id,
    as_of_date: str,
    game_pk,
    ballpark_id: str,
    umpire_id: int = 0,
    catcher_id: int = 0,
    pitcher_hand: str = "R",
    team_id=None,
    standings_gb: float = 5.0,
    schedule_data=None,
) -> dict:
    """
    Compute all 40 features for a single pitcher-date.

    Args:
        pitches:          Statcast pitch-level DataFrame.
        game_logs:        Per-start game log DataFrame.
        lineups:          Projected lineup DataFrame (may be empty).
        weather_df:       Weather DataFrame keyed by game_pk (may be empty).
        umpire_history:   Umpire K tendency data (DataFrame or dict).
        pitcher_id:       MLBAM pitcher ID.
        opponent_team_id: MLBAM team ID for the opposing lineup.
        as_of_date:       Prediction date 'YYYY-MM-DD' (features use data < this date).
        game_pk:          MLB game primary key.
        ballpark_id:      Team abbreviation or venue identifier.
        umpire_id:        Home plate umpire MLBAM ID (0 if unknown).
        catcher_id:       Starting catcher MLBAM ID (0 if unknown).
        pitcher_hand:     'R' or 'L'.
        team_id:          Pitcher's team ID (for bullpen leash calc).
        standings_gb:     Opponent team's games back from wild card (for importance).
        schedule_data:    Optional schedule DataFrame (for travel fatigue).

    Returns:
        dict of feature_name -> value with all 40 features populated.
    """
    features = {}

    # ── Group 1: Pitcher ability (14 features) ────────────────
    try:
        features.update(compute_all_pitcher_ability(pitches, game_logs, pitcher_id, as_of_date))
    except Exception as exc:
        logger.error(f"pitcher_ability failed for {pitcher_id}: {exc}")
        features.update({col: np.nan for col in FEATURE_COLUMNS[:14]})

    # ── Group 2: Recent form (4 features) ────────────────────
    try:
        features.update(compute_all_recent_form(game_logs, pitcher_id, as_of_date))
    except Exception as exc:
        logger.error(f"recent_form failed for {pitcher_id}: {exc}")
        features.update({col: np.nan for col in FEATURE_COLUMNS[14:18]})

    # ── Group 3: Opponent profile (13 features) ──────────────
    # Derive primary velo from pitcher ability (already computed above)
    primary_velo = features.get("fb_velo_avg", 93.0)
    try:
        features.update(compute_all_opponent_profile(
            pitches=pitches,
            game_logs=game_logs,
            lineup_df=lineups,
            pitcher_id=pitcher_id,
            opp_team_id=opponent_team_id,
            as_of_date=as_of_date,
            game_pk=game_pk,
            pitcher_hand=pitcher_hand,
            primary_velo=primary_velo,
            standings_gb=standings_gb,
            schedule_data=schedule_data,
        ))
    except Exception as exc:
        logger.error(f"opponent_profile failed for {pitcher_id}: {exc}")
        features.update({col: np.nan for col in FEATURE_COLUMNS[18:31]})

    # ── Group 4: Contextual (5 features) ─────────────────────
    try:
        features.update(compute_all_contextual(
            ballpark_id=ballpark_id,
            umpire_history=umpire_history,
            umpire_id=umpire_id,
            as_of_date=as_of_date,
            weather_df=weather_df,
            game_pk=game_pk,
            game_logs=game_logs,
            team_id=team_id if team_id is not None else 0,
        ))
    except Exception as exc:
        logger.error(f"contextual failed for {pitcher_id}: {exc}")
        features.update({col: np.nan for col in FEATURE_COLUMNS[31:36]})

    # ── Group 5: Battery effects (4 features) ────────────────
    try:
        features.update(compute_all_battery_effects(
            pitches=pitches,
            game_logs=game_logs,
            lineups=lineups,
            pitcher_id=pitcher_id,
            catcher_id=catcher_id,
            as_of_date=as_of_date,
        ))
    except Exception as exc:
        logger.error(f"battery_effects failed for {pitcher_id}: {exc}")
        features.update({col: np.nan for col in FEATURE_COLUMNS[36:40]})

    # ── Imputation ────────────────────────────────────────────
    features = impute_features(features)

    return features


def build_feature_matrix(
    prediction_date: str,
    starters_df: pd.DataFrame,
    pitches: pd.DataFrame = None,
    game_logs: pd.DataFrame = None,
    lineups: pd.DataFrame = None,
    weather_df: pd.DataFrame = None,
    umpire_history=None,
    schedule_data=None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the full 40-column feature matrix for all probable starters on a given date.

    Args:
        prediction_date: 'YYYY-MM-DD' — features use data strictly before this date.
        starters_df:     DataFrame with columns:
                           pitcher_id, opponent_team_id, game_pk, ballpark_id,
                           pitcher_hand (optional), home_plate_umpire_id (optional),
                           catcher_id (optional), team_id (optional),
                           standings_gb (optional).
        pitches:         Statcast pitch data. Loaded from staging if None.
        game_logs:       Game log data. Loaded from staging if None.
        lineups:         Projected lineup DataFrame (empty DataFrame if None).
        weather_df:      Weather DataFrame keyed by game_pk (empty if None).
        umpire_history:  Umpire tendency data (loaded automatically if None).
        schedule_data:   Schedule data for travel fatigue (None = skip).
        save:            Whether to persist the matrix to features/pitcher_features/{date}.parquet.

    Returns:
        DataFrame with 40 feature columns + metadata columns (pitcher_id, game_date,
        game_pk, opponent_team_id). Empty DataFrame if no data is available.
    """
    # ── Load staging data if not provided ────────────────────
    if pitches is None or game_logs is None:
        _pitches, _games = load_staging_data()
        if pitches is None:
            pitches = _pitches
        if game_logs is None:
            game_logs = _games

    if pitches.empty or game_logs.empty:
        logger.warning(f"No staging data available for {prediction_date}; returning empty matrix")
        return pd.DataFrame()

    if lineups is None:
        lineups = pd.DataFrame()
    if weather_df is None:
        weather_df = pd.DataFrame()
    if umpire_history is None:
        umpire_history = _load_umpire_history()

    # ── Process each probable starter ────────────────────────
    rows = []
    for _, starter in starters_df.iterrows():
        pitcher_id = starter.get("pitcher_id", 0)
        if not pitcher_id or pitcher_id == 0:
            logger.debug("Skipping starter row with missing pitcher_id")
            continue

        logger.info(f"Computing features for pitcher {pitcher_id} | date={prediction_date}")

        try:
            features = compute_features_for_pitcher(
                pitches=pitches,
                game_logs=game_logs,
                lineups=lineups,
                weather_df=weather_df,
                umpire_history=umpire_history,
                pitcher_id=int(pitcher_id),
                opponent_team_id=starter.get("opponent_team_id", 0),
                as_of_date=prediction_date,
                game_pk=starter.get("game_pk", 0),
                ballpark_id=str(starter.get("ballpark_id", "")),
                umpire_id=int(starter.get("home_plate_umpire_id", 0)),
                catcher_id=int(starter.get("catcher_id", 0)),
                pitcher_hand=str(starter.get("pitcher_hand", "R")),
                team_id=starter.get("team_id"),
                standings_gb=float(starter.get("standings_gb", 5.0)),
                schedule_data=schedule_data,
            )
        except Exception as exc:
            logger.error(f"Feature computation failed for pitcher {pitcher_id}: {exc}")
            continue

        # Add metadata
        features["pitcher_id"] = int(pitcher_id)
        features["game_date"] = prediction_date
        features["game_pk"] = starter.get("game_pk", 0)
        features["opponent_team_id"] = starter.get("opponent_team_id", 0)

        rows.append(features)

    if not rows:
        logger.warning(f"No feature rows built for {prediction_date}")
        return pd.DataFrame()

    # ── Assemble DataFrame ────────────────────────────────────
    df = pd.DataFrame(rows)

    # Enforce column order: features first, then metadata
    ordered_cols = (
        [c for c in FEATURE_COLUMNS if c in df.columns] +
        [c for c in METADATA_COLUMNS if c in df.columns]
    )
    df = df[ordered_cols]

    # ── Log fill quality ──────────────────────────────────────
    feature_df = df[[c for c in FEATURE_COLUMNS if c in df.columns]]
    total_cells = feature_df.size
    missing_cells = feature_df.isna().sum().sum()
    fill_rate = 100.0 * (1 - missing_cells / total_cells) if total_cells > 0 else 0.0
    logger.info(
        f"Feature matrix for {prediction_date}: {len(df)} pitchers | "
        f"fill rate={fill_rate:.1f}%"
    )

    # ── Persist to disk ───────────────────────────────────────
    if save:
        out_path = FEATURES_DIR / "pitcher_features" / f"{prediction_date}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved feature matrix -> {out_path}")

    return df


def load_feature_matrix(prediction_date: str) -> pd.DataFrame:
    """
    Load a previously saved feature matrix for a given date.
    Returns empty DataFrame if not found.
    """
    path = FEATURES_DIR / "pitcher_features" / f"{prediction_date}.parquet"
    if not path.exists():
        logger.warning(f"No feature matrix found for {prediction_date} at {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def feature_matrix_exists(prediction_date: str) -> bool:
    """Check whether a feature matrix has already been computed for a date."""
    path = FEATURES_DIR / "pitcher_features" / f"{prediction_date}.parquet"
    return path.exists()
