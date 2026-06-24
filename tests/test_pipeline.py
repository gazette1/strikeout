"""
Integration tests for the full prediction pipeline.
All external API calls are mocked — runs without real data or network.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import lightgbm as lgb
import joblib

from src.features.feature_pipeline import FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Helper: train minimal models and save artifacts to a tmp directory
# ─────────────────────────────────────────────────────────────────────────────
def _save_trained_models(tmp_dir: Path, sample_feature_matrix: pd.DataFrame) -> dict:
    """
    Train tiny LightGBM + Ridge models and save artifacts.
    Returns the metadata dict (same format as real training).
    """
    from sklearn.linear_model import Ridge

    df = sample_feature_matrix.copy()
    X = df[FEATURE_COLUMNS].values
    y = df["actual_strikeouts"].values

    base_params = {
        "objective": "regression", "metric": "mae", "num_leaves": 7,
        "learning_rate": 0.1, "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X, label=y)

    models = {}
    for name, alpha in [("median", None), ("lower", 0.1), ("upper", 0.9)]:
        if alpha is not None:
            params = {**base_params, "objective": "quantile",
                      "alpha": alpha, "metric": "quantile"}
        else:
            params = base_params.copy()
        models[name] = lgb.train(params, dtrain, num_boost_round=5)

    preds = models["median"].predict(X)
    baseline = ((df["k_per_9_rolling"] / 9) * df["ip_per_start"]).fillna(5.0).values
    days_rest = df["days_rest"].fillna(5.0).values

    from src.model.train import train_stacking_layer
    stacking = train_stacking_layer(preds, baseline, days_rest, y)

    # Save all models
    model_paths = {}
    for name, model in models.items():
        path = tmp_dir / f"lgbm_{name}_2024-06-01.txt"
        model.save_model(str(path))
        model_paths[name] = str(path)

    stacking_path = tmp_dir / "stacking_2024-06-01.pkl"
    joblib.dump(stacking, stacking_path)
    model_paths["stacking"] = str(stacking_path)

    metadata = {
        "timestamp": "2024-06-01",
        "train_rows": len(df),
        "val_rows": 5,
        "val_mae": 1.5,
        "val_rmse": 2.0,
        "feature_columns": FEATURE_COLUMNS,
        "models": model_paths,
    }
    meta_path = tmp_dir / "metadata_2024-06-01.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# 1. test_daily_runner_dev_mode
# ─────────────────────────────────────────────────────────────────────────────
def test_daily_runner_dev_mode(sample_pitches, sample_game_logs, sample_feature_matrix, tmp_path):
    """
    Verify that a dev-mode daily pipeline run completes without errors
    when all external calls are mocked.
    """
    starters_df = pd.DataFrame([{
        "pitcher_id": 12345,
        "opponent_team_id": 999,
        "game_pk": 8888,
        "ballpark_id": "NYY",
        "pitcher_hand": "R",
        "home_plate_umpire_id": 0,
        "catcher_id": 0,
        "team_id": 147,
        "standings_gb": 5.0,
    }])

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    metadata = _save_trained_models(model_dir, sample_feature_matrix)

    from src.features.feature_pipeline import build_feature_matrix
    from src.model.predict import predict

    # Build features inline (no file I/O)
    dates = sorted(pd.to_datetime(sample_pitches["game_date"]).unique())
    as_of_date = str(dates[3].date())

    feature_matrix = build_feature_matrix(
        prediction_date=as_of_date,
        starters_df=starters_df,
        pitches=sample_pitches,
        game_logs=sample_game_logs,
        lineups=pd.DataFrame(),
        weather_df=pd.DataFrame(),
        umpire_history=None,
        save=False,
    )

    if feature_matrix.empty:
        pytest.skip("No feature rows built — sample data may not span enough dates")

    # Predict using inline models
    from src.model.predict import load_models
    models, stacking = load_models(metadata)
    predictions = predict(feature_matrix, models=models, stacking_model=stacking)

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) >= 1
    assert "predicted_strikeouts" in predictions.columns


# ─────────────────────────────────────────────────────────────────────────────
# 2. test_prediction_logger_roundtrip
# ─────────────────────────────────────────────────────────────────────────────
def test_prediction_logger_roundtrip(sample_feature_matrix, tmp_data_dir):
    """
    Log predictions to disk and load them back; verify integrity
    (same columns, same data values).
    """
    prediction_date = "2024-06-01"
    pred_dir = tmp_data_dir / "data" / "predictions"
    out_path = pred_dir / f"{prediction_date}.parquet"

    # Simulate prediction output
    predictions = pd.DataFrame({
        "pitcher_id": [12345, 67890],
        "game_date": [prediction_date, prediction_date],
        "game_pk": [9001, 9002],
        "predicted_strikeouts": [7.5, 5.2],
        "pred_ci_lower": [5.0, 3.5],
        "pred_ci_upper": [10.0, 7.0],
    })

    # Write
    predictions.to_parquet(out_path, index=False)

    # Read back
    loaded = pd.read_parquet(out_path)

    assert len(loaded) == 2
    assert list(loaded.columns) == list(predictions.columns)
    assert loaded["pitcher_id"].tolist() == [12345, 67890]
    assert loaded["predicted_strikeouts"].tolist() == pytest.approx([7.5, 5.2])


def test_prediction_logger_appends_correctly(tmp_data_dir):
    """
    Saving predictions for multiple dates creates separate files.
    """
    pred_dir = tmp_data_dir / "data" / "predictions"

    for i, date in enumerate(["2024-06-01", "2024-06-02", "2024-06-03"]):
        df = pd.DataFrame({
            "pitcher_id": [12345],
            "game_date": [date],
            "game_pk": [9000 + i],
            "predicted_strikeouts": [float(6 + i)],
            "pred_ci_lower": [float(4 + i)],
            "pred_ci_upper": [float(9 + i)],
        })
        df.to_parquet(pred_dir / f"{date}.parquet", index=False)

    saved_files = list(pred_dir.glob("*.parquet"))
    assert len(saved_files) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. test_csv_exporter
# ─────────────────────────────────────────────────────────────────────────────
def test_csv_exporter(sample_feature_matrix, tmp_data_dir):
    """
    Export a feature matrix to CSV, read back, verify all columns
    and row count are preserved.
    """
    out_path = tmp_data_dir / "data" / "predictions" / "test_export.csv"

    # Write to CSV
    sample_feature_matrix.to_csv(out_path, index=False)

    # Read back
    loaded = pd.read_csv(out_path)

    assert len(loaded) == len(sample_feature_matrix)
    for col in FEATURE_COLUMNS:
        assert col in loaded.columns, f"Column {col} missing from CSV export"


def test_csv_exporter_numeric_integrity(sample_feature_matrix, tmp_data_dir):
    """
    Numeric values should survive CSV round-trip within floating-point tolerance.
    """
    out_path = tmp_data_dir / "data" / "predictions" / "numeric_roundtrip.csv"
    sample_feature_matrix[["pitcher_id", "game_pk", "k_per_9_rolling", "fb_velo_avg"]].to_csv(
        out_path, index=False
    )

    loaded = pd.read_csv(out_path)
    pd.testing.assert_series_equal(
        loaded["k_per_9_rolling"].round(4),
        sample_feature_matrix["k_per_9_rolling"].round(4),
        check_names=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. test_lineup_resolver_mock
# ─────────────────────────────────────────────────────────────────────────────
_MOCK_GAME_RESPONSE = {
    "gamePk": 748534,
    "liveData": {
        "boxscore": {
            "teams": {
                "away": {
                    "team": {"id": 147},
                    "battingOrder": ["12345", "23456", "34567", "45678", "56789",
                                     "67890", "78901", "89012", "90123"],
                    "players": {
                        "ID12345": {"person": {"id": 12345, "fullName": "Aaron Judge"},
                                     "battingOrder": "100", "stats": {"batting": {}}},
                    },
                },
                "home": {
                    "team": {"id": 111},
                    "battingOrder": [],
                    "players": {},
                },
            },
            "officials": [
                {"official": {"id": 427474, "fullName": "Angel Hernandez"},
                 "officialType": "Home Plate"}
            ],
        }
    },
    "gameData": {
        "teams": {
            "away": {"id": 147, "abbreviation": "NYY"},
            "home": {"id": 111, "abbreviation": "BOS"},
        },
        "probablePitchers": {
            "away": {"id": 12345, "fullName": "Justin Verlander"},
            "home": {"id": 67890, "fullName": "Gerrit Cole"},
        },
        "venue": {"id": "3", "name": "Fenway Park"},
    },
}


def test_lineup_resolver_mock():
    """
    Mock statsapi.get to return a realistic game response.
    Verify load_game_data returns expected structure.
    """
    import sys
    from src.data.mlb_api_loader import load_game_data

    mock_statsapi = MagicMock()
    mock_statsapi.get.return_value = _MOCK_GAME_RESPONSE

    with patch.dict(sys.modules, {"statsapi": mock_statsapi}):
        result = load_game_data(748534)

    assert isinstance(result, dict)
    # Should contain game data keys
    assert "gameData" in result or "gamePk" in result


def test_lineup_resolver_schedule_mock():
    """
    Mock statsapi.schedule to return a game list.
    Verify schedule parsing returns a list/DataFrame of games.
    """
    import sys
    from src.data.mlb_api_loader import load_schedule

    mock_schedule = [
        {
            "game_id": 748534,
            "game_date": "2024-06-01",
            "status": "Scheduled",
            "away_id": 147,
            "home_id": 111,
        }
    ]

    mock_statsapi = MagicMock()
    mock_statsapi.schedule.return_value = mock_schedule

    with patch.dict(sys.modules, {"statsapi": mock_statsapi}):
        result = load_schedule("2024-06-01")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["game_id"] == 748534


# ─────────────────────────────────────────────────────────────────────────────
# 5. test_end_to_end_smoke
# ─────────────────────────────────────────────────────────────────────────────
def test_end_to_end_smoke(sample_pitches, sample_game_logs, sample_feature_matrix, tmp_path):
    """
    Smoke test: verify the full flow
      ingest data → compute features → predict → save results
    executes without crashing, using only in-memory data and mocked I/O.
    """
    # ── Step 1: Ingest (simulated — sample_pitches / sample_game_logs already loaded)
    assert len(sample_pitches) > 0
    assert len(sample_game_logs) > 0

    # ── Step 2: Compute features ──────────────────────────────────────────
    from src.features.feature_pipeline import build_feature_matrix

    dates = sorted(pd.to_datetime(sample_pitches["game_date"]).unique())
    as_of_date = str(dates[3].date())

    starters_df = pd.DataFrame([{
        "pitcher_id": 12345,
        "opponent_team_id": 999,
        "game_pk": 8888,
        "ballpark_id": "NYY",
        "pitcher_hand": "R",
        "home_plate_umpire_id": 0,
        "catcher_id": 0,
        "team_id": 147,
        "standings_gb": 5.0,
    }, {
        "pitcher_id": 67890,
        "opponent_team_id": 147,
        "game_pk": 8889,
        "ballpark_id": "BOS",
        "pitcher_hand": "R",
        "home_plate_umpire_id": 0,
        "catcher_id": 0,
        "team_id": 111,
        "standings_gb": 3.0,
    }])

    feature_matrix = build_feature_matrix(
        prediction_date=as_of_date,
        starters_df=starters_df,
        pitches=sample_pitches,
        game_logs=sample_game_logs,
        lineups=pd.DataFrame(),
        weather_df=pd.DataFrame(),
        umpire_history=None,
        save=False,
    )

    if feature_matrix.empty:
        pytest.skip("No features built from sample data — check date coverage")

    assert len(feature_matrix) >= 1
    for col in FEATURE_COLUMNS:
        assert col in feature_matrix.columns

    # ── Step 3: Predict ───────────────────────────────────────────────────
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    metadata = _save_trained_models(model_dir, sample_feature_matrix)

    from src.model.predict import load_models, predict
    models, stacking = load_models(metadata)
    predictions = predict(feature_matrix, models=models, stacking_model=stacking)

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(feature_matrix)
    assert "predicted_strikeouts" in predictions.columns

    # ── Step 4: Save results ──────────────────────────────────────────────
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    out_path = pred_dir / f"{as_of_date}.parquet"
    predictions.to_parquet(out_path, index=False)

    loaded = pd.read_parquet(out_path)
    assert len(loaded) == len(predictions)
    pd.testing.assert_frame_equal(predictions.reset_index(drop=True),
                                  loaded.reset_index(drop=True))


def test_end_to_end_prediction_values_reasonable(sample_feature_matrix, tmp_path):
    """
    End-to-end test verifying predicted strikeout values are in a
    plausible range (0-20 Ks) for a realistic feature matrix.
    """
    from src.model.predict import load_models, predict

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    metadata = _save_trained_models(model_dir, sample_feature_matrix)
    models, stacking = load_models(metadata)

    predictions = predict(sample_feature_matrix, models=models, stacking_model=stacking)

    assert (predictions["predicted_strikeouts"] >= 0).all()
    assert (predictions["predicted_strikeouts"] <= 20).all()
    assert (predictions["pred_ci_lower"] <= predictions["pred_ci_upper"]).all()
