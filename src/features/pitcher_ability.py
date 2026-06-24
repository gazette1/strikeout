"""
Pitcher Ability features (1-14).
Computed from Statcast pitch-level data and game logs.
"""
import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    LEAGUE_AVG, SWINGING_STRIKE_DESCS, CALLED_STRIKE_DESCS,
    CSW_DESCS, FASTBALL_TYPES, HIGH_K_PITCH_TYPES, SWING_DESCS, GLOBAL_SEED,
)


def exponential_weighted_mean(values: pd.Series, lam: float = 0.92) -> float:
    """
    Compute EWM where the most recent value has highest weight.
    values should be ordered oldest -> newest.
    """
    n = len(values)
    if n == 0:
        return np.nan
    weights = np.array([lam ** (n - 1 - i) for i in range(n)])
    return float(np.average(values.values, weights=weights))


def compute_vaa(vy0, vz0, ay, az, y0=50.0, y_plate=17.0/12.0):
    """
    Vertical Approach Angle at the plate.
    Solve quadratic for time, then compute angle.
    """
    a_coeff = 0.5 * ay
    b_coeff = vy0
    c_coeff = y0 - y_plate
    discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
    if discriminant < 0:
        return np.nan
    t = (-b_coeff - np.sqrt(discriminant)) / (2 * a_coeff)
    if t <= 0:
        return np.nan
    vz_plate = vz0 + az * t
    vy_plate = vy0 + ay * t
    vaa = np.degrees(np.arctan2(vz_plate, -vy_plate))
    return vaa


def compute_vaa_vectorized(df: pd.DataFrame) -> pd.Series:
    """Compute VAA for each pitch row in a DataFrame."""
    required = ["vy0", "vz0", "ay", "az"]
    if not all(c in df.columns for c in required):
        return pd.Series(np.nan, index=df.index)

    mask = df[required].notna().all(axis=1)
    result = pd.Series(np.nan, index=df.index)

    if mask.sum() == 0:
        return result

    sub = df.loc[mask, required]
    a_coeff = 0.5 * sub["ay"]
    b_coeff = sub["vy0"]
    c_coeff = 50.0 - 17.0 / 12.0
    disc = b_coeff**2 - 4 * a_coeff * c_coeff
    valid = disc >= 0

    t = pd.Series(np.nan, index=sub.index)
    t[valid] = (-b_coeff[valid] - np.sqrt(disc[valid])) / (2 * a_coeff[valid])
    t_valid = t > 0

    vz_plate = sub["vz0"] + sub["az"] * t
    vy_plate = sub["vy0"] + sub["ay"] * t
    vaa = np.degrees(np.arctan2(vz_plate, -vy_plate))

    result.loc[mask] = vaa
    result.loc[mask & ~t_valid] = np.nan
    return result


def _get_pitcher_pitches(pitches: pd.DataFrame, pitcher_id: int,
                         as_of_date: str) -> pd.DataFrame:
    """Filter pitches for a given pitcher before the as_of_date."""
    mask = (
        (pitches["pitcher"] == pitcher_id) &
        (pd.to_datetime(pitches["game_date"]) < pd.to_datetime(as_of_date))
    )
    return pitches.loc[mask].copy()


def _get_last_n_starts_pitches(pitches: pd.DataFrame, pitcher_id: int,
                                as_of_date: str, n_starts: int = 3) -> pd.DataFrame:
    """Get pitches from a pitcher's last N starts before as_of_date."""
    pp = _get_pitcher_pitches(pitches, pitcher_id, as_of_date)
    if pp.empty:
        return pp

    game_dates = sorted(pp["game_date"].unique(), reverse=True)[:n_starts]
    return pp[pp["game_date"].isin(game_dates)]


def _get_season_pitches(pitches: pd.DataFrame, pitcher_id: int,
                        as_of_date: str) -> pd.DataFrame:
    """Get pitches from the current season only."""
    pp = _get_pitcher_pitches(pitches, pitcher_id, as_of_date)
    if pp.empty:
        return pp
    year = pd.to_datetime(as_of_date).year
    pp["game_date_dt"] = pd.to_datetime(pp["game_date"])
    return pp[pp["game_date_dt"].dt.year == year]


# ── Feature 1: K/9 rolling (EWM) ──────────────────────────────
def feat_k_per_9_rolling(game_logs: pd.DataFrame, pitcher_id: int,
                         as_of_date: str, lam: float = 0.92) -> float:
    """Season K/9 with exponential weighting. Blueprint Feature 1."""
    gl = game_logs[
        (game_logs["pitcher_id"] == pitcher_id) &
        (pd.to_datetime(game_logs["game_date"]) < pd.to_datetime(as_of_date))
    ].sort_values("game_date")

    year = pd.to_datetime(as_of_date).year
    gl = gl[pd.to_datetime(gl["game_date"]).dt.year == year]

    if len(gl) < 1:
        return LEAGUE_AVG["k_per_9"]

    k9_values = (gl["strikeouts"] / gl["innings_pitched"].clip(lower=0.1)) * 9
    return exponential_weighted_mean(k9_values, lam)


# ── Feature 2: SwStr% ──────────────────────────────────────────
def feat_swstr_pct(pitches: pd.DataFrame, pitcher_id: int,
                   as_of_date: str, window: int = 200) -> float:
    """Swinging strike rate over last N pitches. Blueprint Feature 2."""
    pp = _get_pitcher_pitches(pitches, pitcher_id, as_of_date)
    if pp.empty:
        return LEAGUE_AVG["swstr_pct"]

    recent = pp.sort_values(["game_date", "at_bat_number", "pitch_number"]).tail(window)
    if len(recent) < 50:
        # Expand window
        recent = pp.sort_values(["game_date", "at_bat_number", "pitch_number"]).tail(500)

    if len(recent) == 0:
        return LEAGUE_AVG["swstr_pct"]

    ss_count = recent["description"].isin(SWINGING_STRIKE_DESCS).sum()
    return float(ss_count / len(recent))


# ── Feature 3: CSW% ────────────────────────────────────────────
def feat_csw_pct(pitches: pd.DataFrame, pitcher_id: int,
                 as_of_date: str, window: int = 300) -> float:
    """Called + swinging strike rate. Blueprint Feature 3."""
    pp = _get_pitcher_pitches(pitches, pitcher_id, as_of_date)
    if pp.empty:
        return LEAGUE_AVG["csw_pct"]

    recent = pp.sort_values(["game_date", "at_bat_number", "pitch_number"]).tail(window)
    if len(recent) < 50:
        recent = pp.sort_values(["game_date", "at_bat_number", "pitch_number"]).tail(500)

    if len(recent) == 0:
        return LEAGUE_AVG["csw_pct"]

    csw_count = recent["description"].isin(CSW_DESCS).sum()
    return float(csw_count / len(recent))


# ── Feature 4: Pitch mix K profile ─────────────────────────────
def feat_pitch_mix_k_profile(pitches: pd.DataFrame, pitcher_id: int,
                             as_of_date: str,
                             league_whiff_rates: dict = None) -> float:
    """
    Weighted sum of pitch usage × league-avg whiff rate for high-K types.
    Blueprint Feature 4.
    """
    # Default league whiff rates by pitch type
    if league_whiff_rates is None:
        league_whiff_rates = {
            "SL": 0.33, "FS": 0.35, "ST": 0.38, "CH": 0.30,
            "CU": 0.28, "KC": 0.30, "SV": 0.36, "FF": 0.22,
            "SI": 0.18, "FC": 0.24,
        }

    pp = _get_season_pitches(pitches, pitcher_id, as_of_date)
    if pp.empty:
        return 0.25  # Neutral default

    total = len(pp)
    profile = 0.0
    for pt, whiff in league_whiff_rates.items():
        usage = (pp["pitch_type"] == pt).sum() / total
        profile += usage * whiff

    return float(profile)


# ── Feature 5: FB velo avg (last 3 starts) ─────────────────────
def feat_fb_velo_avg(pitches: pd.DataFrame, pitcher_id: int,
                     as_of_date: str, n_starts: int = 3) -> float:
    """Mean fastball velocity over last N starts. Blueprint Feature 5."""
    pp = _get_last_n_starts_pitches(pitches, pitcher_id, as_of_date, n_starts)
    fb = pp[pp["pitch_type"].isin(FASTBALL_TYPES)]

    if len(fb) < 10:
        # Fall back to season
        pp_season = _get_season_pitches(pitches, pitcher_id, as_of_date)
        fb = pp_season[pp_season["pitch_type"].isin(FASTBALL_TYPES)]

    if fb.empty or fb["release_speed"].isna().all():
        return 93.0  # League average fastball velo

    return float(fb["release_speed"].mean())


# ── Feature 6: FB velo max ─────────────────────────────────────
def feat_fb_velo_max(pitches: pd.DataFrame, pitcher_id: int,
                     as_of_date: str, n_starts: int = 3) -> float:
    """Max fastball velocity over last N starts. Blueprint Feature 6."""
    pp = _get_last_n_starts_pitches(pitches, pitcher_id, as_of_date, n_starts)
    fb = pp[pp["pitch_type"].isin(FASTBALL_TYPES)]

    if fb.empty or fb["release_speed"].isna().all():
        pp_season = _get_season_pitches(pitches, pitcher_id, as_of_date)
        fb = pp_season[pp_season["pitch_type"].isin(FASTBALL_TYPES)]

    if fb.empty or fb["release_speed"].isna().all():
        return 96.0

    return float(fb["release_speed"].max())


# ── Feature 7: FB spin rate ────────────────────────────────────
def feat_fb_spin_rate(pitches: pd.DataFrame, pitcher_id: int,
                      as_of_date: str, n_starts: int = 3) -> float:
    """Mean four-seam spin rate over last N starts. Blueprint Feature 7."""
    pp = _get_last_n_starts_pitches(pitches, pitcher_id, as_of_date, n_starts)
    ff = pp[pp["pitch_type"] == "FF"]

    if ff.empty or ff["release_spin_rate"].isna().all():
        return LEAGUE_AVG["fb_spin_rate"]

    return float(ff["release_spin_rate"].mean())


# ── Feature 8: VAA fastball ────────────────────────────────────
def feat_vaa_fastball(pitches: pd.DataFrame, pitcher_id: int,
                      as_of_date: str, n_starts: int = 3) -> float:
    """Mean VAA for four-seam fastballs. Blueprint Feature 8."""
    pp = _get_last_n_starts_pitches(pitches, pitcher_id, as_of_date, n_starts)
    ff = pp[pp["pitch_type"] == "FF"].copy()

    if ff.empty:
        return -5.0  # Typical FF VAA midpoint

    ff["vaa"] = compute_vaa_vectorized(ff)
    valid = ff["vaa"].dropna()

    if valid.empty:
        return -5.0

    return float(valid.mean())


# ── Feature 9: IVB delta ───────────────────────────────────────
def feat_ivb_delta(pitches: pd.DataFrame, pitcher_id: int,
                   as_of_date: str, league_avg_pfx_z: float = 1.28) -> float:
    """Pitcher's FF IVB minus league average. Blueprint Feature 9."""
    pp = _get_season_pitches(pitches, pitcher_id, as_of_date)
    ff = pp[pp["pitch_type"] == "FF"]

    if ff.empty or ff["pfx_z"].isna().all():
        return 0.0

    return float(ff["pfx_z"].mean() - league_avg_pfx_z)


# ── Feature 10: HB delta ──────────────────────────────────────
def feat_hb_delta(pitches: pd.DataFrame, pitcher_id: int,
                  as_of_date: str, league_avg_pfx_x: float = -0.72) -> float:
    """Pitcher's FF horizontal break minus league average. Feature 10."""
    pp = _get_season_pitches(pitches, pitcher_id, as_of_date)
    ff = pp[pp["pitch_type"] == "FF"]

    if ff.empty or ff["pfx_x"].isna().all():
        return 0.0

    return float(ff["pfx_x"].mean() - league_avg_pfx_x)


# ── Feature 11: Release point consistency ──────────────────────
def feat_release_point_consistency(pitches: pd.DataFrame, pitcher_id: int,
                                    as_of_date: str, n_starts: int = 3) -> float:
    """Std dev of release point across recent starts. Lower = more consistent. Feature 11."""
    pp = _get_last_n_starts_pitches(pitches, pitcher_id, as_of_date, n_starts)

    if pp.empty:
        return 0.15  # League median approximation

    x_std = pp["release_pos_x"].std() if pp["release_pos_x"].notna().sum() > 5 else 0.15
    z_std = pp["release_pos_z"].std() if pp["release_pos_z"].notna().sum() > 5 else 0.15

    return float(np.sqrt(x_std**2 + z_std**2))


# ── Feature 12: Arm slot drift ─────────────────────────────────
def feat_arm_slot_drift(pitches: pd.DataFrame, pitcher_id: int,
                        as_of_date: str, n_starts: int = 3) -> float:
    """Euclidean distance of recent release point from season baseline. Feature 12."""
    pp_season = _get_season_pitches(pitches, pitcher_id, as_of_date)
    pp_recent = _get_last_n_starts_pitches(pitches, pitcher_id, as_of_date, n_starts)

    if pp_season.empty or pp_recent.empty:
        return 0.0

    season_x = pp_season["release_pos_x"].mean()
    season_z = pp_season["release_pos_z"].mean()
    recent_x = pp_recent["release_pos_x"].mean()
    recent_z = pp_recent["release_pos_z"].mean()

    if any(pd.isna([season_x, season_z, recent_x, recent_z])):
        return 0.0

    return float(np.sqrt((recent_x - season_x)**2 + (recent_z - season_z)**2))


# ── Feature 13: Tunneling efficiency ──────────────────────────
def feat_tunneling_efficiency(pitches: pd.DataFrame, pitcher_id: int,
                              as_of_date: str, n_starts: int = 3) -> float:
    """
    Simplified V1: plate separation minus release separation between
    FF and best secondary pitch (SL or CH). Blueprint Feature 13.
    """
    pp = _get_last_n_starts_pitches(pitches, pitcher_id, as_of_date, n_starts)

    if pp.empty:
        return 0.5  # League median approximation

    ff = pp[pp["pitch_type"] == "FF"]

    # Find best secondary (most used non-fastball)
    non_fb = pp[~pp["pitch_type"].isin(FASTBALL_TYPES)]
    if non_fb.empty or ff.empty:
        return 0.5

    secondary_type = non_fb["pitch_type"].value_counts().index[0]
    sec = pp[pp["pitch_type"] == secondary_type]

    if sec.empty:
        return 0.5

    # Release separation
    rel_dx = abs(ff["release_pos_x"].mean() - sec["release_pos_x"].mean())
    rel_dz = abs(ff["release_pos_z"].mean() - sec["release_pos_z"].mean())
    rel_sep = np.sqrt(rel_dx**2 + rel_dz**2) if not any(pd.isna([rel_dx, rel_dz])) else 0

    # Plate separation
    plate_dx = abs(ff["plate_x"].mean() - sec["plate_x"].mean())
    plate_dz = abs(ff["plate_z"].mean() - sec["plate_z"].mean())
    plate_sep = np.sqrt(plate_dx**2 + plate_dz**2) if not any(pd.isna([plate_dx, plate_dz])) else 0

    # Higher = better tunneling (large plate sep from small release sep)
    efficiency = plate_sep - rel_sep if (plate_sep + rel_sep) > 0 else 0.0
    return float(max(efficiency, 0.0))


# ── Feature 14: Two-strike put-away rate ───────────────────────
def feat_two_strike_putaway_rate(pitches: pd.DataFrame, pitcher_id: int,
                                  as_of_date: str, lam: float = 0.90) -> float:
    """K per PA reaching 2-strike count. Blueprint Feature 14."""
    pp = _get_season_pitches(pitches, pitcher_id, as_of_date)

    if pp.empty:
        return LEAGUE_AVG["putaway_rate"]

    # PAs that reached 2 strikes
    two_strike = pp[(pp["strikes"] == 2) & pp["events"].notna()]

    if len(two_strike) < 5:
        return LEAGUE_AVG["putaway_rate"]

    strikeouts = (two_strike["events"] == "strikeout").sum()
    return float(strikeouts / len(two_strike))


def compute_all_pitcher_ability(pitches: pd.DataFrame, game_logs: pd.DataFrame,
                                 pitcher_id: int, as_of_date: str) -> dict:
    """Compute all 14 pitcher ability features. Returns dict of feature_name -> value."""
    return {
        "k_per_9_rolling": feat_k_per_9_rolling(game_logs, pitcher_id, as_of_date),
        "swstr_pct": feat_swstr_pct(pitches, pitcher_id, as_of_date),
        "csw_pct": feat_csw_pct(pitches, pitcher_id, as_of_date),
        "pitch_mix_k_profile": feat_pitch_mix_k_profile(pitches, pitcher_id, as_of_date),
        "fb_velo_avg": feat_fb_velo_avg(pitches, pitcher_id, as_of_date),
        "fb_velo_max": feat_fb_velo_max(pitches, pitcher_id, as_of_date),
        "fb_spin_rate": feat_fb_spin_rate(pitches, pitcher_id, as_of_date),
        "vaa_fastball": feat_vaa_fastball(pitches, pitcher_id, as_of_date),
        "ivb_delta": feat_ivb_delta(pitches, pitcher_id, as_of_date),
        "hb_delta": feat_hb_delta(pitches, pitcher_id, as_of_date),
        "release_point_consistency": feat_release_point_consistency(pitches, pitcher_id, as_of_date),
        "arm_slot_drift": feat_arm_slot_drift(pitches, pitcher_id, as_of_date),
        "tunneling_efficiency": feat_tunneling_efficiency(pitches, pitcher_id, as_of_date),
        "two_strike_putaway_rate": feat_two_strike_putaway_rate(pitches, pitcher_id, as_of_date),
    }
