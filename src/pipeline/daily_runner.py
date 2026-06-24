"""
daily_runner.py
---------------
Main daily prediction pipeline.
Run at ~6 AM ET daily via cron/GitHub Actions.

Usage:
    python -m src.pipeline.daily_runner [--date YYYY-MM-DD] [--dev]

Dev mode: uses a fixed past date range for testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from config.settings import (
    PREDICTIONS_DIR,
    RAW_LINEUPS,
)

from src.data.statcast_loader import load_statcast_day
from src.data.mlb_api_loader import extract_pitcher_game_logs, load_schedule
from src.data.weather_loader import load_weather_for_date
from src.features.feature_pipeline import build_feature_matrix
from src.model.predict import predict
from src.pipeline.lineup_resolver import get_probable_starters
from src.pipeline import prediction_logger
from src.pipeline import csv_exporter


# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

DEV_DATE = "2024-07-15"


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------

def run_daily_pipeline(
    prediction_date: str = None,
    dev_mode: bool = False,
) -> pd.DataFrame:
    """
    Execute every stage of the daily strikeout-prediction pipeline.

    Stages
    ------
    1. **Ingest** — Pull yesterday's Statcast pitches, pitcher game logs,
       and weather data.
    2. **Resolve Starters** — Fetch today's probable starters via the
       lineup resolver.
    3. **Features** — Build the 40-column feature matrix.
    4. **Predict** — Run inference with the production model ensemble.
    5. **Log & Export** — Persist predictions and write a CSV export.
    6. **Score Prior** — Score yesterday's predictions against actual
       results.

    Each stage is wrapped in its own ``try/except`` so a single failure
    does not abort the full run.

    Parameters
    ----------
    prediction_date : str, optional
        ``YYYY-MM-DD`` date for which to generate predictions.
        Defaults to today.
    dev_mode : bool
        When ``True``, forces ``prediction_date = DEV_DATE`` for local
        testing against known historical data.

    Returns
    -------
    pd.DataFrame
        Predictions DataFrame (may be empty if inference fails).
    """
    # ── Resolve date ──────────────────────────────────────────────────────
    if dev_mode:
        prediction_date = DEV_DATE
    elif prediction_date is None:
        prediction_date = date.today().isoformat()

    yesterday = (
        date.fromisoformat(prediction_date) - timedelta(days=1)
    ).isoformat()

    logger.info(f"Starting daily pipeline for {prediction_date}")

    # Shared state passed between stages
    statcast_df: pd.DataFrame = pd.DataFrame()
    game_logs_df: pd.DataFrame = pd.DataFrame()
    weather_df: pd.DataFrame = pd.DataFrame()
    starters_df: pd.DataFrame = pd.DataFrame()
    feature_matrix: pd.DataFrame = pd.DataFrame()
    predictions: pd.DataFrame = pd.DataFrame()

    # ── Step 1 — INGEST ───────────────────────────────────────────────────
    logger.info("Step 1 — Ingesting yesterday's data")

    try:
        statcast_df = load_statcast_day(yesterday)
        logger.info(
            f"Loaded {len(statcast_df):,} Statcast pitches for {yesterday}"
        )
    except Exception as exc:
        logger.error(f"Statcast ingest failed for {yesterday}: {exc}")

    try:
        game_logs_df = extract_pitcher_game_logs(yesterday)
        logger.info(
            f"Extracted {len(game_logs_df)} pitcher game logs for {yesterday}"
        )
    except Exception as exc:
        logger.error(f"Pitcher game log extraction failed for {yesterday}: {exc}")

    try:
        schedule = load_schedule(prediction_date)
        schedule_df = pd.DataFrame(schedule) if schedule else pd.DataFrame()
        weather_df = load_weather_for_date(prediction_date, schedule_df)
        logger.info(
            f"Loaded weather for {len(weather_df)} games on {prediction_date}"
        )
    except Exception as exc:
        logger.warning(f"Weather fetch failed for {prediction_date}: {exc}")

    # ── Step 2 — RESOLVE STARTERS ─────────────────────────────────────────
    logger.info("Step 2 — Resolving probable starters")

    try:
        starters_df = get_probable_starters(prediction_date)
        logger.info(
            f"Found {len(starters_df)} probable starters for {prediction_date}"
        )
    except Exception as exc:
        logger.error(f"Starter resolution failed for {prediction_date}: {exc}")

    if starters_df.empty:
        logger.warning(
            "No probable starters found — pipeline will continue but predictions "
            "may be empty."
        )

    # ── Step 3 — FEATURES ─────────────────────────────────────────────────
    logger.info("Step 3 — Building feature matrix")

    try:
        feature_matrix = build_feature_matrix(
            prediction_date=prediction_date,
            starters_df=starters_df,
            pitches=statcast_df if not statcast_df.empty else None,
            game_logs=game_logs_df if not game_logs_df.empty else None,
            weather_df=weather_df if not weather_df.empty else None,
        )
        logger.info(
            f"Feature matrix shape: {feature_matrix.shape} for {prediction_date}"
        )
    except Exception as exc:
        logger.error(f"Feature pipeline failed for {prediction_date}: {exc}")

    # ── Step 4 — PREDICT ──────────────────────────────────────────────────
    logger.info("Step 4 — Running inference")

    try:
        if feature_matrix.empty:
            logger.warning(
                "Feature matrix is empty — skipping inference for "
                f"{prediction_date}"
            )
        else:
            predictions = predict(feature_matrix)
            logger.info(
                f"Generated {len(predictions)} predictions for {prediction_date}"
            )
    except FileNotFoundError as exc:
        logger.warning(
            f"No trained model found — skipping inference. "
            f"Train a model first. Details: {exc}"
        )
    except Exception as exc:
        logger.error(f"Inference failed for {prediction_date}: {exc}")

    # ── Step 5 — LOG & EXPORT ─────────────────────────────────────────────
    logger.info("Step 5 — Logging and exporting predictions")

    try:
        if not predictions.empty:
            prediction_logger.log_predictions(predictions, prediction_date)
            csv_exporter.export_predictions(predictions, prediction_date)
            logger.info(f"Predictions saved for {prediction_date}")
        else:
            logger.warning(
                f"No predictions to log/export for {prediction_date}"
            )
    except Exception as exc:
        logger.error(f"Log/export step failed for {prediction_date}: {exc}")

    # ── Step 6 — SCORE PRIOR ──────────────────────────────────────────────
    logger.info("Step 6 — Scoring yesterday's predictions")

    try:
        scored = prediction_logger.score_predictions(yesterday)
        if not scored.empty:
            metrics = prediction_logger.compute_rolling_accuracy(scored)
            logger.info(
                f"Yesterday's scoring complete — "
                f"MAE={metrics.get('mae', 'N/A'):.3f}, "
                f"RMSE={metrics.get('rmse', 'N/A'):.3f}, "
                f"O/U Acc={metrics.get('ou_accuracy', 'N/A'):.3f}"
            )
        else:
            logger.warning(
                f"No prior predictions found to score for {yesterday}"
            )
    except Exception as exc:
        logger.error(f"Scoring step failed for {yesterday}: {exc}")

    logger.info(f"Daily pipeline complete for {prediction_date}")
    return predictions


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line interface for the daily prediction pipeline.

    Flags
    -----
    --date YYYY-MM-DD
        Override the prediction date (default: today).
    --dev
        Enable dev mode — uses a fixed historical date for local testing.
    """
    parser = argparse.ArgumentParser(
        description="MLB strikeout daily prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline.daily_runner\n"
            "  python -m src.pipeline.daily_runner --date 2024-08-01\n"
            "  python -m src.pipeline.daily_runner --dev\n"
        ),
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Prediction date (default: today)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help=f"Dev mode: use fixed date {DEV_DATE} for testing",
    )
    args = parser.parse_args()

    run_daily_pipeline(prediction_date=args.date, dev_mode=args.dev)


if __name__ == "__main__":
    main()
