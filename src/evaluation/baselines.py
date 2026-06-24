"""
Baseline prediction models for K-Predictor benchmarking.

Three baselines from the project blueprint:
  B1 — Season K/9 average × expected IP
  B2 — Vegas proxy: season_k9 × team_avg_ip × park_factor / 100
  B3 — Last-5-start K rolling average

Each function accepts a DataFrame with the required columns and returns
a 1-D numpy array of predicted strikeouts, one entry per row.

Required columns per baseline:
  B1: k_per_9_season (or k_per_9_rolling), ip_per_start
  B2: k_per_9_season (or k_per_9_rolling), team_avg_ip, park_k_factor
  B3: k_last_5_avg  (pre-computed rolling feature)

Fallback to league-average constants from config.settings.LEAGUE_AVG when
any column is missing or null.
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUE_AVG


# ── helpers ───────────────────────────────────────────────────────────────────

def _series(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    """Return column values as float array, filling missing with *default*."""
    if col in df.columns:
        return df[col].fillna(default).astype(float).values
    return np.full(len(df), default, dtype=float)


# ── B1: Season K/9 × Expected IP ─────────────────────────────────────────────

def baseline_k9_x_ip(df: pd.DataFrame) -> np.ndarray:
    """
    B1 — Season K/9 × IP-per-start baseline.

    Formula:
        predicted_k = (k_per_9 / 9) × ip_per_start

    Preferred columns (falls back to league averages if absent):
        k_per_9_season  → first choice
        k_per_9_rolling → second choice
        ip_per_start    → expected IP for the start
    """
    # Prefer season K/9; fall back to rolling, then league avg
    if "k_per_9_season" in df.columns:
        k9 = df["k_per_9_season"].fillna(LEAGUE_AVG["k_per_9"]).astype(float).values
    else:
        k9 = _series(df, "k_per_9_rolling", LEAGUE_AVG["k_per_9"])

    ip = _series(df, "ip_per_start", LEAGUE_AVG["ip_per_start"])

    return (k9 / 9.0) * ip


# ── B2: Vegas proxy ───────────────────────────────────────────────────────────

def baseline_vegas_proxy(df: pd.DataFrame) -> np.ndarray:
    """
    B2 — Vegas-style proxy using park factor adjustment.

    Formula:
        predicted_k = (k_per_9 / 9) × team_avg_ip × (park_k_factor / 100)

    Columns:
        k_per_9_season / k_per_9_rolling  — pitcher K rate
        team_avg_ip                       — team's average IP allowed per start
        park_k_factor                     — park strikeout factor (100 = neutral)
    """
    if "k_per_9_season" in df.columns:
        k9 = df["k_per_9_season"].fillna(LEAGUE_AVG["k_per_9"]).astype(float).values
    else:
        k9 = _series(df, "k_per_9_rolling", LEAGUE_AVG["k_per_9"])

    team_ip = _series(df, "team_avg_ip", LEAGUE_AVG["ip_per_start"])
    park_factor = _series(df, "park_k_factor", LEAGUE_AVG["park_k_factor"])

    return (k9 / 9.0) * team_ip * (park_factor / 100.0)


# ── B3: Last-5-start K average ────────────────────────────────────────────────

def baseline_last5_avg(df: pd.DataFrame) -> np.ndarray:
    """
    B3 — Rolling average of actual Ks over the pitcher's last 5 starts.

    Expects column `k_last_5_avg` to be pre-computed in the feature matrix.
    Falls back to the league-average K/9 × IP if the column is absent or null.
    """
    fallback = LEAGUE_AVG["k_per_9"] / 9.0 * LEAGUE_AVG["ip_per_start"]
    return _series(df, "k_last_5_avg", fallback)


# ── convenience wrapper ───────────────────────────────────────────────────────

def all_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all three baselines and return them as a DataFrame aligned to *df*.

    Returns columns: baseline_b1, baseline_b2, baseline_b3
    """
    return pd.DataFrame(
        {
            "baseline_b1": baseline_k9_x_ip(df),
            "baseline_b2": baseline_vegas_proxy(df),
            "baseline_b3": baseline_last5_avg(df),
        },
        index=df.index,
    )
