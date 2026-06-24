"""
prediction_logger.py
--------------------
Handles persistence, scoring, and accuracy reporting for MLB strikeout
predictions.

Functions
---------
log_predictions(predictions, date)
    Persist a DataFrame of predictions to the daily predictions store.

score_predictions(date) -> pd.DataFrame
    Join predictions for *date* against actual results and compute errors.

load_prediction_history(n_days) -> pd.DataFrame
    Load and concatenate the last *n_days* of scored predictions.

compute_rolling_accuracy(scored_df) -> dict
    Compute MAE, RMSE, and over/under accuracy from a scored DataFrame.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import PREDICTIONS_DIR, STAGING_GAMES, RAW_GAME_LOGS


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _daily_dir() -> Path:
    """Return (and create) the directory for raw daily predictions."""
    p = PREDICTIONS_DIR / "daily"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _eval_dir() -> Path:
    """Return (and create) the directory for scored evaluation files."""
    p = PREDICTIONS_DIR / "evaluation"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# log_predictions
# ---------------------------------------------------------------------------

def log_predictions(predictions: pd.DataFrame, date_str: str) -> None:
    """
    Persist a predictions DataFrame to the daily predictions store.

    The file is written as Parquet to
    ``{PREDICTIONS_DIR}/daily/{date}.parquet``.  A ``logged_at`` timestamp
    column is attached before writing.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output from ``src.model.predict.predict``.  Expected columns:
        ``pitcher_id``, ``game_date``, ``game_pk``,
        ``predicted_strikeouts``, ``pred_ci_lower``, ``pred_ci_upper``.
    date_str : str
        Prediction date string ``YYYY-MM-DD``.  Used as the file name.

    Raises
    ------
    ValueError
        If *predictions* is empty.
    """
    if predictions.empty:
        raise ValueError(
            f"Cannot log empty predictions DataFrame for {date_str}"
        )

    out_path = _daily_dir() / f"{date_str}.parquet"

    df = predictions.copy()
    df["logged_at"] = pd.Timestamp.utcnow()

    df.to_parquet(out_path, index=False)
    logger.info(
        f"Logged {len(df)} predictions for {date_str} → {out_path}"
    )


# ---------------------------------------------------------------------------
# score_predictions
# ---------------------------------------------------------------------------

def _load_actuals(date_str: str) -> pd.DataFrame:
    """
    Load actual pitcher results for *date_str*.

    Tries ``RAW_GAME_LOGS/{date_str}.parquet`` first, then falls back to
    ``STAGING_GAMES`` (filtered by game_date).  Returns an empty DataFrame
    when neither source is available.
    """
    # Preferred: per-date game log cache (from mlb_api_loader)
    raw_path = RAW_GAME_LOGS / f"{date_str}.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        logger.debug(f"Loaded actuals from {raw_path}")
        return df

    # Fallback: staging games table
    if STAGING_GAMES.exists():
        staging = pd.read_parquet(STAGING_GAMES)
        if "game_date" in staging.columns:
            mask = staging["game_date"].astype(str).str.startswith(date_str)
            filtered = staging[mask]
            if not filtered.empty:
                logger.debug(
                    f"Loaded {len(filtered)} actuals from staging for {date_str}"
                )
                return filtered

    logger.warning(f"No actuals found for {date_str}")
    return pd.DataFrame()


def score_predictions(date_str: str) -> pd.DataFrame:
    """
    Score predictions for *date_str* against actual results.

    Loads the stored predictions and the actual game outcomes, joins them
    on ``(game_pk, pitcher_id)``, and computes the per-prediction
    residual ``error = actual_strikeouts - predicted_strikeouts``.

    The scored DataFrame is written to
    ``{PREDICTIONS_DIR}/evaluation/{date_str}_scored.parquet``.

    Parameters
    ----------
    date_str : str
        Date string ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        Scored predictions with columns including ``actual_strikeouts``
        and ``error``.  Returns an empty DataFrame when predictions or
        actuals are unavailable.
    """
    pred_path = _daily_dir() / f"{date_str}.parquet"
    if not pred_path.exists():
        logger.warning(f"No predictions file for {date_str}: {pred_path}")
        return pd.DataFrame()

    predictions = pd.read_parquet(pred_path)

    actuals = _load_actuals(date_str)
    if actuals.empty:
        logger.warning(
            f"Actuals unavailable for {date_str} — cannot score predictions"
        )
        return pd.DataFrame()

    # Normalise join keys
    if "strikeouts" in actuals.columns:
        actuals = actuals.rename(columns={"strikeouts": "actual_strikeouts"})
    elif "actual_strikeouts" not in actuals.columns:
        logger.warning(
            f"Actuals table for {date_str} has no 'strikeouts' column"
        )
        return pd.DataFrame()

    join_cols = [c for c in ("game_pk", "pitcher_id") if c in actuals.columns]
    if not join_cols:
        logger.warning(
            f"Actuals table for {date_str} missing join keys (game_pk / pitcher_id)"
        )
        return pd.DataFrame()

    scored = predictions.merge(
        actuals[join_cols + ["actual_strikeouts"]],
        on=join_cols,
        how="inner",
    )

    if scored.empty:
        logger.warning(
            f"No rows matched between predictions and actuals for {date_str}"
        )
        return pd.DataFrame()

    scored["error"] = scored["actual_strikeouts"] - scored["predicted_strikeouts"]
    scored["abs_error"] = scored["error"].abs()
    scored["scored_date"] = date_str

    out_path = _eval_dir() / f"{date_str}_scored.parquet"
    scored.to_parquet(out_path, index=False)
    logger.info(
        f"Scored {len(scored)} predictions for {date_str} "
        f"(MAE={scored['abs_error'].mean():.3f}) → {out_path}"
    )

    return scored


# ---------------------------------------------------------------------------
# load_prediction_history
# ---------------------------------------------------------------------------

def load_prediction_history(n_days: int = 30) -> pd.DataFrame:
    """
    Load and concatenate the last *n_days* of scored predictions.

    Iterates backwards from yesterday looking for
    ``{PREDICTIONS_DIR}/evaluation/{date}_scored.parquet`` files.
    Missing dates are silently skipped.

    Parameters
    ----------
    n_days : int
        Number of calendar days to look back (default 30).

    Returns
    -------
    pd.DataFrame
        Concatenated scored predictions, or an empty DataFrame when no
        history files are found.
    """
    today = date.today()
    frames: list[pd.DataFrame] = []

    for delta in range(1, n_days + 1):
        target_date = (today - timedelta(days=delta)).isoformat()
        scored_path = _eval_dir() / f"{target_date}_scored.parquet"
        if scored_path.exists():
            try:
                frames.append(pd.read_parquet(scored_path))
            except Exception as exc:
                logger.warning(
                    f"Could not read scored file {scored_path}: {exc}"
                )

    if not frames:
        logger.warning(f"No scored prediction history found in the last {n_days} days")
        return pd.DataFrame()

    history = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Loaded {len(history)} scored predictions across {len(frames)} days"
    )
    return history


# ---------------------------------------------------------------------------
# compute_rolling_accuracy
# ---------------------------------------------------------------------------

def compute_rolling_accuracy(scored_df: pd.DataFrame) -> dict:
    """
    Compute aggregate accuracy metrics from a scored predictions DataFrame.

    Metrics
    -------
    mae : float
        Mean absolute error.
    rmse : float
        Root mean squared error.
    ou_accuracy : float
        Fraction of predictions where the over/under sign matches
        (``error >= 0`` means the actual exceeded the prediction).
    n : int
        Number of scored predictions.

    Parameters
    ----------
    scored_df : pd.DataFrame
        DataFrame containing at minimum ``error`` and ``abs_error`` columns
        (produced by :func:`score_predictions`).

    Returns
    -------
    dict
        Keys: ``mae``, ``rmse``, ``ou_accuracy``, ``n``.
        Returns a dict of NaN values when *scored_df* is empty.
    """
    if scored_df.empty or "error" not in scored_df.columns:
        logger.warning("Cannot compute rolling accuracy — scored_df is empty or missing 'error' column")
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "ou_accuracy": float("nan"),
            "n": 0,
        }

    errors = scored_df["error"].dropna()
    n = len(errors)

    if n == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "ou_accuracy": float("nan"),
            "n": 0,
        }

    mae: float = float(errors.abs().mean())
    rmse: float = float(np.sqrt((errors ** 2).mean()))
    ou_accuracy: float = float((errors >= 0).mean())

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "ou_accuracy": round(ou_accuracy, 4),
        "n": n,
    }

    logger.info(
        f"Rolling accuracy ({n} predictions) — "
        f"MAE={mae:.3f}, RMSE={rmse:.3f}, O/U Acc={ou_accuracy:.3f}"
    )
    return metrics
