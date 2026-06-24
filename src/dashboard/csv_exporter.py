"""
csv_exporter.py
---------------
Export prediction and evaluation data to CSV files for the MLB K-Predictor
dashboard layer.

Exports:
  - Daily prediction snapshots  →  data/predictions/daily/{date}_predictions.csv
  - Per-game evaluation summaries → data/predictions/evaluation/{date}_summary.csv
  - Weekly roll-up reports        → data/predictions/evaluation/weekly_{s}_{e}.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import PREDICTIONS_DIR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _resolve_output_dir(output_dir: Optional[Path], subdir: str) -> Path:
    """Return *output_dir* if provided, else fall back to PREDICTIONS_DIR/subdir."""
    if output_dir is not None:
        return Path(output_dir)
    return PREDICTIONS_DIR / subdir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_daily_predictions(
    predictions: pd.DataFrame,
    date: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Export today's pitcher predictions to a dated CSV file.

    Parameters
    ----------
    predictions:
        DataFrame with at minimum the columns: ``pitcher_id``, ``game_date``,
        ``game_pk``, ``predicted_strikeouts``, ``pred_ci_lower``,
        ``pred_ci_upper``.
    date:
        ISO-8601 date string used in the output filename, e.g. ``"2026-04-15"``.
    output_dir:
        Destination directory.  Defaults to
        ``PREDICTIONS_DIR / "daily"``.

    Returns
    -------
    Path
        Absolute path of the written CSV file.

    Raises
    ------
    ValueError
        If *predictions* is empty or missing required columns.
    """
    required_cols = {
        "pitcher_id", "game_date", "game_pk",
        "predicted_strikeouts", "pred_ci_lower", "pred_ci_upper",
    }
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions DataFrame is missing columns: {missing}")
    if predictions.empty:
        raise ValueError("predictions DataFrame must not be empty.")

    dest_dir = _resolve_output_dir(output_dir, "daily")
    _ensure_dir(dest_dir)

    out_path = dest_dir / f"{date}_predictions.csv"

    df = predictions[sorted(required_cols)].copy()
    df["exported_at"] = datetime.now(timezone.utc).isoformat()

    # Normalise column order for readability
    col_order = [
        "pitcher_id", "game_date", "game_pk",
        "predicted_strikeouts", "pred_ci_lower", "pred_ci_upper",
        "exported_at",
    ]
    df = df[col_order]

    df.to_csv(out_path, index=False)
    logger.info(f"Exported {len(df)} daily predictions → {out_path}")
    return out_path


def export_evaluation_summary(
    scored_df: pd.DataFrame,
    date: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Export per-pitcher evaluation scores and aggregate summary to CSV.

    The function expects *scored_df* to contain both predicted and actual
    strikeout columns.  It computes per-pitcher absolute errors and appends
    summary statistics as trailing rows.

    Parameters
    ----------
    scored_df:
        DataFrame with columns ``pitcher_id``, ``predicted_strikeouts``,
        ``actual_strikeouts`` (plus optional CI columns).
    date:
        ISO-8601 date string used in the output filename.
    output_dir:
        Destination directory.  Defaults to
        ``PREDICTIONS_DIR / "evaluation"``.

    Returns
    -------
    Path
        Absolute path of the written CSV file.

    Raises
    ------
    ValueError
        If required columns are absent or DataFrame is empty.
    """
    required_cols = {"pitcher_id", "predicted_strikeouts", "actual_strikeouts"}
    missing = required_cols - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df is missing columns: {missing}")
    if scored_df.empty:
        raise ValueError("scored_df must not be empty.")

    dest_dir = _resolve_output_dir(output_dir, "evaluation")
    _ensure_dir(dest_dir)

    out_path = dest_dir / f"{date}_summary.csv"

    df = scored_df.copy()
    df["abs_error"] = (df["predicted_strikeouts"] - df["actual_strikeouts"]).abs()
    df["signed_error"] = df["predicted_strikeouts"] - df["actual_strikeouts"]

    # Optional CI columns
    ci_cols = [c for c in ("pred_ci_lower", "pred_ci_upper", "game_date", "game_pk")
               if c in df.columns]
    keep_cols = ["pitcher_id"] + ci_cols + [
        "predicted_strikeouts", "actual_strikeouts", "abs_error", "signed_error"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].sort_values("abs_error", ascending=False)

    # Summary statistics rows
    summary_rows = {
        "SUMMARY_MAE":  df["abs_error"].mean(),
        "SUMMARY_RMSE": (df["signed_error"] ** 2).mean() ** 0.5,
        "SUMMARY_BIAS": df["signed_error"].mean(),
        "SUMMARY_N":    len(df),
        "SUMMARY_MAX_ERROR": df["abs_error"].max(),
        "SUMMARY_MIN_ERROR": df["abs_error"].min(),
    }
    summary_df = pd.DataFrame([
        {"pitcher_id": k, "predicted_strikeouts": v}
        for k, v in summary_rows.items()
    ])

    final_df = pd.concat([df, summary_df], ignore_index=True)
    final_df.to_csv(out_path, index=False)
    logger.info(
        f"Exported evaluation summary ({len(df)} pitchers, "
        f"MAE={summary_rows['SUMMARY_MAE']:.3f}) → {out_path}"
    )
    return out_path


def export_weekly_report(
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Aggregate all daily predictions and actuals over a week and export a CSV.

    The function scans ``PREDICTIONS_DIR/daily/`` for all files whose dates
    fall within [start_date, end_date], loads them, and computes aggregate
    metrics.  If evaluation CSVs exist for the same period they are merged to
    provide MAE / RMSE figures; otherwise the report contains prediction
    columns only.

    Parameters
    ----------
    start_date:
        Inclusive start of the weekly window, e.g. ``"2026-04-14"``.
    end_date:
        Inclusive end of the weekly window, e.g. ``"2026-04-20"``.
    output_dir:
        Destination directory.  Defaults to
        ``PREDICTIONS_DIR / "evaluation"``.

    Returns
    -------
    Path
        Absolute path of the written CSV file.

    Raises
    ------
    FileNotFoundError
        If no daily prediction files are found for the given date range.
    """
    dest_dir = _resolve_output_dir(output_dir, "evaluation")
    _ensure_dir(dest_dir)

    daily_dir = PREDICTIONS_DIR / "daily"
    eval_dir = PREDICTIONS_DIR / "evaluation"

    # ---- Gather daily prediction files in the window -----------------------
    start_dt = pd.Timestamp(start_date).normalize()
    end_dt = pd.Timestamp(end_date).normalize()

    pred_frames: list[pd.DataFrame] = []
    if daily_dir.exists():
        for fp in sorted(daily_dir.glob("*_predictions.csv")):
            try:
                file_date = pd.Timestamp(fp.stem.replace("_predictions", ""))
                if start_dt <= file_date <= end_dt:
                    pred_frames.append(pd.read_csv(fp))
            except Exception as exc:
                logger.warning(f"Skipping {fp.name}: {exc}")

    if not pred_frames:
        raise FileNotFoundError(
            f"No daily prediction files found in '{daily_dir}' "
            f"for {start_date} → {end_date}."
        )

    pred_df = pd.concat(pred_frames, ignore_index=True)

    # ---- Gather evaluation summary files -----------------------------------
    eval_frames: list[pd.DataFrame] = []
    if eval_dir.exists():
        for fp in sorted(eval_dir.glob("*_summary.csv")):
            try:
                file_date = pd.Timestamp(fp.stem.replace("_summary", ""))
                if start_dt <= file_date <= end_dt:
                    eval_frames.append(pd.read_csv(fp))
            except Exception as exc:
                logger.warning(f"Skipping eval file {fp.name}: {exc}")

    # ---- Build the report --------------------------------------------------
    report_rows = []

    # Aggregate prediction stats per pitcher
    for pid, grp in pred_df.groupby("pitcher_id"):
        row: dict = {
            "pitcher_id": pid,
            "n_games": len(grp),
            "avg_predicted_k": grp["predicted_strikeouts"].mean(),
            "min_predicted_k": grp["predicted_strikeouts"].min(),
            "max_predicted_k": grp["predicted_strikeouts"].max(),
        }
        if "pred_ci_lower" in grp.columns and "pred_ci_upper" in grp.columns:
            row["avg_ci_width"] = (
                grp["pred_ci_upper"] - grp["pred_ci_lower"]
            ).mean()
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)

    # Merge evaluation data if available
    if eval_frames:
        eval_df = pd.concat(eval_frames, ignore_index=True)
        # Filter out summary rows (pitcher_id starts with "SUMMARY_")
        eval_df = eval_df[~eval_df["pitcher_id"].astype(str).str.startswith("SUMMARY_")]

        if "abs_error" in eval_df.columns:
            eval_agg = eval_df.groupby("pitcher_id").agg(
                mae=("abs_error", "mean"),
                rmse_approx=("signed_error", lambda x: (x**2).mean()**0.5)
                if "signed_error" in eval_df.columns
                else ("abs_error", "mean"),
            ).reset_index()
            report_df = report_df.merge(eval_agg, on="pitcher_id", how="left")

    # Sort by worst MAE first (if available), else by pitcher_id
    if "mae" in report_df.columns:
        report_df = report_df.sort_values("mae", ascending=False)
    else:
        report_df = report_df.sort_values("pitcher_id")

    # Append a weekly summary row
    summary: dict = {
        "pitcher_id": "WEEKLY_TOTAL",
        "n_games": report_df["n_games"].sum(),
        "avg_predicted_k": report_df["avg_predicted_k"].mean(),
        "min_predicted_k": report_df["min_predicted_k"].min(),
        "max_predicted_k": report_df["max_predicted_k"].max(),
    }
    if "mae" in report_df.columns:
        summary["mae"] = report_df["mae"].mean()
    if "rmse_approx" in report_df.columns:
        summary["rmse_approx"] = report_df["rmse_approx"].mean()

    report_df = pd.concat(
        [report_df, pd.DataFrame([summary])], ignore_index=True
    )

    out_path = dest_dir / f"weekly_{start_date}_{end_date}.csv"
    report_df.to_csv(out_path, index=False)
    logger.info(
        f"Exported weekly report ({start_date} → {end_date}, "
        f"{len(report_df)-1} pitchers) → {out_path}"
    )
    return out_path
