"""
Global configuration for the MLB K-Predictor system.
All paths, seeds, and system-wide constants live here.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = DATA_DIR / "models"

# ── Data paths ────────────────────────────────────────────────
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"

RAW_STATCAST = RAW_DIR / "statcast"
RAW_GAME_LOGS = RAW_DIR / "game_logs"
RAW_LINEUPS = RAW_DIR / "lineups"
RAW_UMPIRES = RAW_DIR / "umpires"
RAW_WEATHER = RAW_DIR / "weather"

STAGING_PITCHES = STAGING_DIR / "pitches.parquet"
STAGING_GAMES = STAGING_DIR / "games.parquet"
STAGING_PLAYERS = STAGING_DIR / "players.parquet"

PRODUCTION_MODEL_DIR = MODEL_DIR / "production"
EXPERIMENT_MODEL_DIR = MODEL_DIR / "experiments"
MODEL_REGISTRY = MODEL_DIR / "metadata" / "model_registry.json"

# ── Reproducibility ───────────────────────────────────────────
GLOBAL_SEED = 42

# ── Rate limiting ─────────────────────────────────────────────
STATCAST_DELAY_SECONDS = 15  # Between weekly backfill requests
MLB_API_DELAY_SECONDS = 1.0  # Courtesy throttle
METEOSTAT_DELAY_SECONDS = 0.5

# ── League averages (updated annually, used as imputation defaults)
LEAGUE_AVG = {
    "k_per_9": 8.5,
    "swstr_pct": 0.115,
    "csw_pct": 0.295,
    "fb_spin_rate": 2250.0,
    "ip_per_start": 5.3,
    "days_rest_default": 5,
    "team_k_pct": 0.225,
    "o_swing_pct": 0.31,
    "z_contact_pct": 0.82,
    "contact_pct": 0.76,
    "putaway_rate": 0.30,
    "lineup_sub_risk": 0.15,
    "park_k_factor": 100,
    "dome_temp_f": 72.0,
    "dome_humidity_pct": 50.0,
}

# ── Lineup slot PA weights (expected PA per lineup position) ──
LINEUP_SLOT_PA_WEIGHTS = {
    1: 4.8, 2: 4.7, 3: 4.5, 4: 4.4, 5: 4.3,
    6: 4.2, 7: 4.1, 8: 3.9, 9: 3.8,
}

# ── Velocity bands for opponent chase-rate feature ────────────
VELO_BANDS = {
    "hard": (95.0, 110.0),
    "medium": (92.0, 95.0),
    "soft": (0.0, 92.0),
}

# ── Statcast description categories ──────────────────────────
SWINGING_STRIKE_DESCS = {"swinging_strike", "swinging_strike_blocked"}
CALLED_STRIKE_DESCS = {"called_strike"}
CSW_DESCS = SWINGING_STRIKE_DESCS | CALLED_STRIKE_DESCS
SWING_DESCS = SWINGING_STRIKE_DESCS | {
    "foul", "foul_tip", "hit_into_play", "foul_bunt",
    "hit_into_play_no_out", "hit_into_play_score",
    "missed_bunt", "swinging_pitchout",
}
FASTBALL_TYPES = {"FF", "SI", "FC"}
HIGH_K_PITCH_TYPES = {"SL", "FS", "ST", "CH", "CU", "KC", "SV"}

# ── High zone IDs for catcher aggression (Statcast zones 1,2,3,11,12)
HIGH_ZONE_IDS = {1, 2, 3, 11, 12}
