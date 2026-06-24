"""
Evaluation metrics for the K prediction system.
"""
import numpy as np
import pandas as pd


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_over_under_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                 line: float) -> float:
    """Fraction of predictions that correctly predict over/under a given K line."""
    pred_side = y_pred > line
    actual_side = y_true > line
    # Exclude pushes (actual == line)
    mask = y_true != line
    if mask.sum() == 0:
        return 0.5
    return float((pred_side[mask] == actual_side[mask]).mean())


def compute_interval_calibration(y_true: np.ndarray, lower: np.ndarray,
                                  upper: np.ndarray) -> float:
    """Fraction of actuals within the 90% prediction interval."""
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def compute_mae_by_tier(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE segmented by actual K tier."""
    tiers = {
        "0-3": (0, 3),
        "4-6": (4, 6),
        "7-9": (7, 9),
        "10+": (10, 100),
    }
    results = {}
    for name, (low, high) in tiers.items():
        mask = (y_true >= low) & (y_true <= high)
        if mask.sum() > 0:
            results[name] = compute_mae(y_true[mask], y_pred[mask])
        else:
            results[name] = None
    return results


def full_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray,
                           lower: np.ndarray = None, upper: np.ndarray = None) -> dict:
    """Generate a full evaluation report."""
    report = {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "n_samples": len(y_true),
        "over_under_accuracy": {},
        "mae_by_tier": compute_mae_by_tier(y_true, y_pred),
    }

    for line in [5.5, 6.5, 7.5, 8.5]:
        report["over_under_accuracy"][str(line)] = compute_over_under_accuracy(y_true, y_pred, line)

    if lower is not None and upper is not None:
        report["interval_calibration_90"] = compute_interval_calibration(y_true, lower, upper)

    return report
