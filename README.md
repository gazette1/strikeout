# MLB Pitcher Strikeout Prediction System

A machine-learning pipeline that generates **daily pitcher strikeout predictions with 90% confidence intervals** for every scheduled MLB starting pitcher. Built on publicly available data sources — no paid subscriptions required.

---

## What It Does

- Pulls Statcast pitch-level data, MLB schedule/lineup data, historical umpire tendencies, and ballpark weather forecasts each morning
- Engineers **40 features** across five groups: pitcher ability, recent form, opponent profile, contextual factors, and battery effects
- Trains three **LightGBM quantile regression** models (5th, 50th, 95th percentiles) stacked with a Ridge meta-learner
- Outputs a prediction file with median K estimate and 90% CI for every starting pitcher before first pitch
- Serves results through an interactive **Streamlit dashboard**

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Dev backfill (July 2024 only, ~15 min)

```bash
make backfill-dev
```

### 3. Predict for a specific date

```bash
make predict DATE=2024-08-01
```

### 4. Full backtest (walk-forward CV)

```bash
make backtest
```

### 5. Run tests

```bash
make test
```

### 6. Launch dashboard

```bash
make dashboard
```

---

## All Make Commands

| Command | Description |
|---|---|
| `make install` | Install package in editable mode with dev deps |
| `make backfill` | Pull full historical data (2022–2025, ~4 hrs) |
| `make backfill-dev` | Pull July 2024 only (~15 min, for development) |
| `make predict DATE=YYYY-MM-DD` | Run daily prediction pipeline for a specific date |
| `make backtest` | Walk-forward cross-validation over held-out seasons |
| `make blind-test` | Blind test on final holdout set |
| `make train` | Train production model on full dataset |
| `make tune` | Run Optuna hyperparameter search (100 trials, 1 hr) |
| `make dashboard` | Launch Streamlit prediction dashboard |
| `make test` | Run pytest suite |
| `make clean` | Remove `__pycache__` and `.pyc` files |

---

## Architecture

### Data Sources (all free)

| Source | Library | Data |
|---|---|---|
| Baseball Savant / Statcast | `pybaseball` | Pitch-level data: velocity, spin, movement, outcomes |
| MLB Stats API | `MLB-StatsAPI` | Schedules, lineups, game logs, umpire assignments |
| Meteostat | `meteostat` | Historical and forecast weather by lat/lon |
| OpenWeatherMap *(optional backup)* | `requests` | Weather fallback (key in `.env`) |

### Feature Groups (40 total)

| Group | Features | Count |
|---|---|---|
| **Pitcher Ability** | K/9 rolling, SwStr%, CSW%, pitch mix K-profile, velo, spin, movement, release consistency, tunneling, putaway rate | 14 |
| **Recent Form** | K rate last 5 starts, pitch count trend, IP/start, days rest | 4 |
| **Opponent Profile** | Team K rate vs. hand, O-swing%, Z-contact%, contact rate, projected lineup K rate, whiff vs. pitch types, chase vs. velo band, handedness stack, sub risk, matchup familiarity, travel fatigue, game importance | 13 |
| **Contextual** | Park K factor, umpire K boost, temperature, humidity, pitcher leash | 5 |
| **Battery Effects** | Catcher framing runs, battery K rate together/delta, catcher game-calling aggression | 4 |

### Model Stack

```
Statcast + MLB API + Weather
         │
         ▼
   Feature Engineering (40 features)
         │
         ▼
┌────────────────────────────────────┐
│  LightGBM Quantile Regression      │
│  ├── q=0.05  (lower bound)         │
│  ├── q=0.50  (median prediction)   │
│  └── q=0.95  (upper bound)         │
└────────────────────────────────────┘
         │
         ▼
   Ridge Meta-Learner (stacking)
         │
         ▼
   Predictions + 90% CI
```

Walk-forward cross-validation is used for all model evaluation — no future data leaks into training windows.

---

## Project Structure

```
mlb-k-predictor/
├── config/
│   ├── settings.py          # Paths, seeds, constants, league averages
│   ├── model_params.yaml    # LightGBM hyperparameters, Optuna search space
│   ├── features.yaml        # All 40 feature definitions and fallback logic
│   └── ballparks.yaml       # All 30 MLB venues (lat/lon, elevation, dome/roof)
├── data/
│   ├── raw/                 # Raw data by source (statcast, game_logs, etc.)
│   ├── staging/             # Cleaned parquet files (pitches, games, players)
│   ├── features/            # Computed feature matrices
│   ├── predictions/         # Daily output files
│   └── models/
│       ├── production/      # Active production model artifacts
│       ├── experiments/     # Experiment snapshots
│       └── metadata/        # model_registry.json
├── src/
│   ├── data/
│   │   ├── backfill.py      # Historical data ingestion (Statcast + MLB API)
│   │   ├── statcast.py      # Statcast fetcher with rate limiting
│   │   ├── mlb_api.py       # MLB Stats API client
│   │   └── weather.py       # Meteostat weather fetcher
│   ├── features/
│   │   ├── pitcher.py       # Pitcher ability features (IDs 1–14)
│   │   ├── recent_form.py   # Recent form features (IDs 15–18)
│   │   ├── opponent.py      # Opponent profile features (IDs 19–31)
│   │   ├── contextual.py    # Park/ump/weather features (IDs 32–36)
│   │   └── battery.py       # Battery effect features (IDs 37–40)
│   ├── model/
│   │   ├── train.py         # Model training entrypoint
│   │   ├── predict.py       # Inference on new feature rows
│   │   └── hyperparameter_tuning.py  # Optuna study
│   ├── evaluation/
│   │   ├── walk_forward.py  # Walk-forward CV backtesting
│   │   └── blind_test.py    # Final holdout evaluation
│   ├── pipeline/
│   │   └── daily_runner.py  # End-to-end daily orchestration
│   └── dashboard/
│       └── streamlit_app.py # Interactive prediction viewer
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_pipeline.py
├── scripts/
│   ├── backfill_all.sh      # Full 2022–2025 historical backfill
│   ├── daily_cron.sh        # Wrapper for cron/scheduled runs
│   └── deploy_model.sh      # Promote experiment model to production
├── .github/
│   └── workflows/
│       └── daily_predict.yml  # GitHub Actions: runs at 6 AM ET daily
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
└── README.md
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `OPENWEATHER_API_KEY` | — | Optional backup weather source |
| `DATA_DIR` | `<project_root>/data` | Override data directory |
| `MODEL_DIR` | `<DATA_DIR>/models` | Override model directory |

### Key Settings (`config/settings.py`)

- `GLOBAL_SEED = 42` — Reproducibility seed used everywhere
- `STATCAST_DELAY_SECONDS = 15` — Pause between Statcast API requests (respect rate limits)
- `LEAGUE_AVG` — Annual league average values used as imputation defaults when a feature cannot be computed from available data

### Model Parameters (`config/model_params.yaml`)

LightGBM quantile models with `num_leaves=63`, `learning_rate=0.05`, early stopping at 50 rounds. Optuna tunes over 100 trials with a 1-hour budget. Ridge stacking layer uses `alpha=1.0`.

### Features (`config/features.yaml`)

Each of the 40 features has a defined computation window, fallback chain, and optional `league_avg_key` for imputation. Edit fallback chains here without touching source code.

### Ballparks (`config/ballparks.yaml`)

All 30 venues with accurate coordinates, elevation, and roof type. Used for weather fetching and park factor assignment.

---

## Dev Mode

For fast iteration, use the dev backfill target which pulls only July 2024 (~15 minutes vs. ~4 hours for the full historical load):

```bash
# Pull one month of data
make backfill-dev

# Then predict and backtest against that window
make predict DATE=2024-07-15
make backtest
```

---

## Automated Daily Runs

The `.github/workflows/daily_predict.yml` workflow runs automatically at **10:00 UTC (6:00 AM ET)** each day, ingests the day's schedule and lineups, generates predictions, uploads them as a build artifact, and commits the output back to the repository.

To trigger manually:

```bash
# Via GitHub CLI
gh workflow run daily_predict.yml

# Or via the GitHub UI: Actions → Daily K Predictions → Run workflow
```

To run locally on a cron schedule:

```bash
# Add to crontab (runs at 6:05 AM ET)
5 6 * * * /path/to/mlb-k-predictor/scripts/daily_cron.sh >> /var/log/mlb-k-predictor.log 2>&1
```
