"""
Shared pytest fixtures for the MLB K-Predictor test suite.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: sample_pitches
# Realistic Statcast pitch-level DataFrame (~200 rows, 2 pitchers, 10 dates)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_pitches():
    rng = np.random.default_rng(42)

    pitcher_ids = [12345, 67890]
    dates = pd.date_range("2024-04-01", periods=10, freq="6D").strftime("%Y-%m-%d").tolist()
    pitch_types = ["FF", "SL", "CH", "CU"]
    descriptions = [
        "swinging_strike", "called_strike", "foul", "hit_into_play",
        "ball", "swinging_strike_blocked", "hit_into_play_no_out",
        "foul_tip", "hit_into_play_score",
    ]
    events_pool = [None, None, None, "strikeout", "field_out", "single", "home_run", "walk"]

    rows = []
    game_pk_counter = 1000
    for pitcher_id in pitcher_ids:
        for date_idx, game_date in enumerate(dates):
            game_pk = game_pk_counter
            game_pk_counter += 1
            n_pitches = rng.integers(15, 25)
            for ab in range(1, 6):
                at_bat_pitches = rng.integers(3, 7)
                for pitch_num in range(1, int(at_bat_pitches) + 1):
                    pitch_type = rng.choice(pitch_types, p=[0.45, 0.25, 0.20, 0.10])
                    desc = rng.choice(descriptions)
                    is_fastball = pitch_type in ("FF", "SI", "FC")
                    velo = float(rng.normal(95.0, 1.5) if is_fastball else rng.normal(84.0, 2.0))
                    spin = float(rng.normal(2280, 150) if pitch_type == "FF" else rng.normal(2400, 200))
                    rel_x = float(rng.normal(-1.5, 0.15))
                    rel_z = float(rng.normal(6.0, 0.12))
                    pfx_x = float(rng.normal(-0.7, 0.3))
                    pfx_z = float(rng.normal(1.3, 0.4))
                    strikes = min(2, int(rng.integers(0, 3)))
                    balls = min(3, int(rng.integers(0, 4)))
                    zone = int(rng.choice(list(range(1, 10)) + [11, 12, 13, 14]))
                    # End-of-AB event only on last pitch
                    event = rng.choice(events_pool) if pitch_num == int(at_bat_pitches) else None

                    rows.append({
                        "pitcher": pitcher_id,
                        "game_date": game_date,
                        "game_pk": game_pk,
                        "pitch_type": str(pitch_type),
                        "description": str(desc),
                        "release_speed": velo,
                        "release_spin_rate": spin,
                        "pfx_x": pfx_x,
                        "pfx_z": pfx_z,
                        "release_pos_x": rel_x,
                        "release_pos_z": rel_z,
                        "plate_x": float(rng.normal(0.0, 0.8)),
                        "plate_z": float(rng.normal(2.5, 0.5)),
                        "zone": zone,
                        "batter": int(rng.integers(100000, 999999)),
                        "stand": rng.choice(["R", "L"]),
                        "p_throws": "R",
                        "events": event,
                        "strikes": strikes,
                        "balls": balls,
                        "at_bat_number": ab,
                        "pitch_number": pitch_num,
                        "inning": int(rng.integers(1, 7)),
                        "vy0": float(rng.normal(-130, 3)),
                        "vz0": float(rng.normal(-5, 2)),
                        "ay": float(rng.normal(24, 2)),
                        "az": float(rng.normal(-18, 2)),
                    })

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: sample_game_logs
# Synthetic per-start game log for 2 pitchers, 5 games each
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_game_logs():
    rng = np.random.default_rng(42)

    pitcher_data = [
        (12345, "Justin Verlander", 137, "NYY"),
        (67890, "Gerrit Cole",      147, "BOS"),
    ]
    dates = pd.date_range("2024-04-01", periods=5, freq="6D").strftime("%Y-%m-%d").tolist()

    rows = []
    game_pk = 5000
    for pitcher_id, name, team_id, opp_abbr in pitcher_data:
        opp_team_id = 999 if opp_abbr == "BOS" else 111
        for i, game_date in enumerate(dates):
            ip = float(rng.choice([4.0, 5.0, 5.2, 6.0, 6.1, 7.0]))
            ks = int(rng.integers(3, 11))
            pitches = int(rng.integers(75, 110))
            rows.append({
                "game_pk": game_pk,
                "pitcher_id": pitcher_id,
                "pitcher_name": name,
                "team_id": team_id,
                "opponent_team_id": opp_team_id,
                "game_date": game_date,
                "innings_pitched": ip,
                "strikeouts": ks,
                "pitches_thrown": pitches,
                "walks": int(rng.integers(0, 4)),
                "earned_runs": int(rng.integers(0, 5)),
                "hits_allowed": int(rng.integers(2, 9)),
                "is_home": bool(i % 2 == 0),
                "ballpark_id": "NYY" if i % 2 == 0 else "BOS",
                "pitcher_hand": "R",
            })
            game_pk += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: sample_lineups
# Synthetic lineup with 9 batters
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_lineups():
    rng = np.random.default_rng(42)
    rows = []
    game_pk = 7777
    team_id = 999
    for slot in range(1, 10):
        rows.append({
            "game_pk": game_pk,
            "team_id": team_id,
            "batter_id": int(rng.integers(200000, 800000)),
            "batting_order": slot,
            "batter_hand": rng.choice(["R", "L"]),
            "start_rate_14d": float(rng.uniform(0.4, 1.0)),
            "games_started_14d": int(rng.integers(5, 14)),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: sample_feature_matrix
# Full 40-feature DataFrame (20 rows) with realistic values + target
# ─────────────────────────────────────────────────────────────────────────────
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

# Per-feature realistic value ranges (lo, hi)
_FEATURE_RANGES = {
    "k_per_9_rolling": (6.0, 13.0),
    "swstr_pct": (0.07, 0.18),
    "csw_pct": (0.23, 0.38),
    "pitch_mix_k_profile": (0.18, 0.35),
    "fb_velo_avg": (89.0, 100.0),
    "fb_velo_max": (91.0, 103.0),
    "fb_spin_rate": (1900.0, 2700.0),
    "vaa_fastball": (-6.5, -3.5),
    "ivb_delta": (-0.5, 0.8),
    "hb_delta": (-0.5, 0.5),
    "release_point_consistency": (0.05, 0.30),
    "arm_slot_drift": (0.0, 0.25),
    "tunneling_efficiency": (0.0, 1.2),
    "two_strike_putaway_rate": (0.20, 0.45),
    "k_rate_last_5": (5.0, 14.0),
    "recent_pitch_count_trend": (-3.0, 3.0),
    "ip_per_start": (4.0, 7.0),
    "days_rest": (4.0, 8.0),
    "opp_team_k_rate_vs_hand": (0.15, 0.30),
    "opp_o_swing_pct": (0.24, 0.38),
    "opp_z_contact_pct": (0.75, 0.92),
    "opp_contact_rate": (0.70, 0.88),
    "projected_lineup_weighted_k_rate": (0.15, 0.30),
    "opp_whiff_vs_pitch_types": (0.10, 0.38),
    "opp_chase_vs_velo_band": (0.24, 0.38),
    "lineup_handedness_stack": (0, 9),
    "opp_plate_discipline_variance": (0.02, 0.10),
    "lineup_sub_risk": (0.05, 0.30),
    "matchup_familiarity": (0, 3),
    "opp_travel_fatigue": (0, 2),
    "opp_game_importance": (0, 2),
    "park_k_factor": (96.0, 103.0),
    "ump_k_boost": (-0.03, 0.04),
    "temperature_f": (50.0, 90.0),
    "humidity_pct": (30.0, 80.0),
    "pitcher_leash": (0.5, 1.5),
    "catcher_framing_runs": (-5.0, 10.0),
    "battery_k_rate_together": (6.0, 13.0),
    "battery_k_rate_delta": (-1.5, 1.5),
    "catcher_game_calling_aggression": (0.20, 0.45),
}


@pytest.fixture
def sample_feature_matrix():
    rng = np.random.default_rng(42)
    n = 20
    data = {}
    for col in FEATURE_COLUMNS:
        lo, hi = _FEATURE_RANGES.get(col, (0.0, 1.0))
        if col in ("lineup_handedness_stack", "matchup_familiarity",
                    "opp_travel_fatigue", "opp_game_importance"):
            data[col] = rng.integers(int(lo), int(hi) + 1, size=n).astype(float)
        else:
            data[col] = rng.uniform(lo, hi, size=n)

    # Metadata
    data["pitcher_id"] = [12345] * 10 + [67890] * 10
    data["game_date"] = (
        pd.date_range("2024-05-01", periods=10, freq="6D").strftime("%Y-%m-%d").tolist() * 2
    )
    data["game_pk"] = list(range(9000, 9020))
    data["opponent_team_id"] = [999] * 20

    # Target
    data["actual_strikeouts"] = rng.integers(1, 12, size=n).astype(float)

    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: tmp_data_dir
# Creates temp directories matching the project data structure
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_data_dir(tmp_path):
    dirs = [
        tmp_path / "data" / "raw" / "statcast",
        tmp_path / "data" / "raw" / "game_logs",
        tmp_path / "data" / "raw" / "lineups",
        tmp_path / "data" / "raw" / "umpires",
        tmp_path / "data" / "raw" / "weather",
        tmp_path / "data" / "staging",
        tmp_path / "data" / "features" / "pitcher_features",
        tmp_path / "data" / "predictions",
        tmp_path / "data" / "models" / "production",
        tmp_path / "data" / "models" / "experiments",
        tmp_path / "data" / "models" / "metadata",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return tmp_path
