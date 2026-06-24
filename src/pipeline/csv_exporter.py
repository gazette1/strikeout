"""
csv_exporter.py
---------------
Thin wrapper around ``src.dashboard.csv_exporter.export_daily_predictions``
that exposes a simple ``export_predictions(predictions, date)`` interface
for use by the daily pipeline runner.

This module lives in the pipeline package so that ``daily_runner`` can do::

    from src.pipeline import csv_exporter
    csv_exporter.export_predictions(predictions, date)

without importing dashboard-layer internals directly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import PREDICTIONS_DIR
from src.dashboard.csv_exporter import export_daily_predictions


def export_predictions(
    predictions: pd.DataFrame,
    date_str: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Export a predictions DataFrame to a dated CSV file.

    Delegates to :func:`src.dashboard.csv_exporter.export_daily_predictions`.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions DataFrame from ``src.model.predict.predict``.
    date_str : str
        Prediction date ``YYYY-MM-DD``.
    output_dir : Path, optional
        Override output directory.  Defaults to
        ``{PREDICTIONS_DIR}/daily``.

    Returns
    -------
    Path
        Path of the written CSV file.
    """
    try:
        out_path = export_daily_predictions(
            predictions=predictions,
            date=date_str,
            output_dir=output_dir,
        )
        logger.info(f"CSV export complete for {date_str} → {out_path}")
        return out_path
    except Exception as exc:
        logger.error(f"CSV export failed for {date_str}: {exc}")
        raise
