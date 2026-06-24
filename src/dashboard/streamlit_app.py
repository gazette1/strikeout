"""
streamlit_app.py
----------------
Streamlit dashboard for the MLB K-Predictor system.

Run with:
    streamlit run src/dashboard/streamlit_app.py

Sections
--------
1. Today's Predictions   – pitcher table for the selected date
2. Model Performance     – rolling MAE / O-U accuracy over last 30 days
3. Feature Importance    – top-15 SHAP or native-gain features
4. Prediction History    – pitcher-level prediction vs actuals
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import datetime
import glob as _glob
from typing import Optional

import pandas as pd
import streamlit as st

from config.settings import (
    PREDICTIONS_DIR,
    PRODUCTION_MODEL_DIR,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLB K-Predictor Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers / cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_daily_predictions(date: str) -> Optional[pd.DataFrame]:
    """Load predictions for *date* from parquet or CSV (whichever exists first)."""
    daily_dir = PREDICTIONS_DIR / "daily"
    for suffix in (f"{date}.parquet", f"{date}_predictions.parquet",
                   f"{date}_predictions.csv", f"{date}.csv"):
        fp = daily_dir / suffix
        if fp.exists():
            try:
                if fp.suffix == ".parquet":
                    return pd.read_parquet(fp)
                return pd.read_csv(fp)
            except Exception as exc:
                st.warning(f"Could not read {fp.name}: {exc}")
    return None


@st.cache_data(ttl=300)
def _load_scored_history(days: int = 30) -> pd.DataFrame:
    """Scan evaluation/ directory and load the last *days* summary CSVs."""
    eval_dir = PREDICTIONS_DIR / "evaluation"
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    frames = []
    if eval_dir.exists():
        for fp in sorted(eval_dir.glob("*_summary.csv")):
            try:
                file_date = pd.Timestamp(fp.stem.replace("_summary", ""))
                if file_date >= cutoff:
                    df = pd.read_csv(fp)
                    df["_eval_date"] = file_date
                    frames.append(df)
            except Exception:
                pass
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        # Drop summary sentinel rows
        return combined[
            ~combined["pitcher_id"].astype(str).str.startswith("SUMMARY_")
        ].copy()
    return pd.DataFrame()


@st.cache_data(ttl=600)
def _load_feature_importance() -> pd.DataFrame:
    """Try to load saved SHAP or gain feature importance from the production model dir."""
    for fname in ("shap_importance.csv", "feature_importance.csv", "gain_importance.csv"):
        fp = PRODUCTION_MODEL_DIR / fname
        if fp.exists():
            try:
                return pd.read_csv(fp)
            except Exception:
                pass
    # Fallback: try JSON
    for fname in ("feature_importance.json",):
        fp = PRODUCTION_MODEL_DIR / fname
        if fp.exists():
            try:
                df = pd.read_json(fp)
                return df
            except Exception:
                pass
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_all_predictions_for_pitcher(pitcher_id: str, n: int = 10) -> pd.DataFrame:
    """Load the last *n* predictions for a specific pitcher across all daily files."""
    daily_dir = PREDICTIONS_DIR / "daily"
    frames = []
    for fp in sorted(daily_dir.glob("*_predictions.csv")):
        try:
            df = pd.read_csv(fp)
            sub = df[df["pitcher_id"].astype(str) == str(pitcher_id)]
            if not sub.empty:
                frames.append(sub)
        except Exception:
            pass
    # Also check parquet
    for fp in sorted(daily_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(fp)
            sub = df[df["pitcher_id"].astype(str) == str(pitcher_id)]
            if not sub.empty:
                frames.append(sub)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "game_date" in combined.columns:
        combined = combined.sort_values("game_date", ascending=False)
    return combined.head(n)


@st.cache_data(ttl=300)
def _load_actuals_for_pitcher(pitcher_id: str) -> pd.DataFrame:
    """Load actual strikeout data from evaluation CSVs for a given pitcher."""
    eval_dir = PREDICTIONS_DIR / "evaluation"
    frames = []
    if eval_dir.exists():
        for fp in sorted(eval_dir.glob("*_summary.csv")):
            try:
                df = pd.read_csv(fp)
                sub = df[df["pitcher_id"].astype(str) == str(pitcher_id)]
                if not sub.empty:
                    date_str = fp.stem.replace("_summary", "")
                    sub = sub.copy()
                    sub["_eval_date"] = date_str
                    frames.append(sub)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "_eval_date" in combined.columns:
        combined = combined.sort_values("_eval_date", ascending=False)
    return combined


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚾ MLB K-Predictor")
    st.markdown("---")

    selected_date = st.date_input(
        "Prediction Date",
        value=datetime.date.today(),
        min_value=datetime.date(2020, 1, 1),
        max_value=datetime.date.today() + datetime.timedelta(days=7),
    )
    date_str = selected_date.strftime("%Y-%m-%d")

    st.markdown("---")
    st.markdown("### Model Info")
    model_info_path = PRODUCTION_MODEL_DIR / "model_registry.json"
    if model_info_path.exists():
        try:
            import json
            with open(model_info_path) as f:
                registry = json.load(f)
            if isinstance(registry, list) and registry:
                latest = registry[-1]
            elif isinstance(registry, dict):
                latest = registry
            else:
                latest = {}
            st.json(latest)
        except Exception as exc:
            st.caption(f"Could not load model registry: {exc}")
    else:
        st.caption("No model registry found.")

    st.markdown("---")
    st.caption(f"Dashboard last refreshed: {datetime.datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("MLB K-Predictor Dashboard")
st.caption(f"Showing data for **{date_str}**")

# ── Section 1: Today's Predictions ──────────────────────────────────────────
st.header("Today's Predictions")

try:
    daily_preds = _load_daily_predictions(date_str)
    if daily_preds is not None and not daily_preds.empty:
        # Display columns in preferred order
        display_cols = [c for c in [
            "pitcher_id", "game_date", "game_pk",
            "predicted_strikeouts", "pred_ci_lower", "pred_ci_upper",
        ] if c in daily_preds.columns]

        col1, col2, col3 = st.columns(3)
        col1.metric("Pitchers", len(daily_preds))
        if "predicted_strikeouts" in daily_preds.columns:
            col2.metric("Avg Predicted K", f"{daily_preds['predicted_strikeouts'].mean():.2f}")
            col3.metric("Max Predicted K", f"{daily_preds['predicted_strikeouts'].max():.0f}")

        st.dataframe(
            daily_preds[display_cols].sort_values(
                "predicted_strikeouts", ascending=False
            ).reset_index(drop=True),
            use_container_width=True,
            height=350,
        )
    else:
        st.info(
            f"No prediction data found for {date_str}.  "
            "Run the prediction pipeline first or select a different date."
        )
except Exception as exc:
    st.error(f"Error loading daily predictions: {exc}")

st.divider()

# ── Section 2: Model Performance ────────────────────────────────────────────
st.header("Model Performance (Last 30 Days)")

try:
    scored_history = _load_scored_history(days=30)
    if scored_history.empty:
        st.info("No scored prediction history found for the last 30 days.")
    else:
        perf_col1, perf_col2 = st.columns(2)

        # Rolling MAE by date
        with perf_col1:
            st.subheader("Rolling MAE by Date")
            if "abs_error" in scored_history.columns and "_eval_date" in scored_history.columns:
                mae_by_date = (
                    scored_history.groupby("_eval_date")["abs_error"]
                    .mean()
                    .reset_index()
                    .rename(columns={"_eval_date": "date", "abs_error": "MAE"})
                    .sort_values("date")
                )
                mae_by_date["date"] = mae_by_date["date"].astype(str)
                st.line_chart(mae_by_date.set_index("date")["MAE"])
            else:
                st.caption("MAE data not available (need `abs_error` column).")

        # Over/Under accuracy by date
        with perf_col2:
            st.subheader("O/U Accuracy by Date (5.5 K Line)")
            if all(c in scored_history.columns
                   for c in ("predicted_strikeouts", "actual_strikeouts", "_eval_date")):
                def _ou_acc(grp: pd.DataFrame, line: float = 5.5) -> float:
                    pred_over = grp["predicted_strikeouts"] > line
                    act_over = grp["actual_strikeouts"] > line
                    mask = grp["actual_strikeouts"] != line
                    if mask.sum() == 0:
                        return 0.5
                    return float((pred_over[mask] == act_over[mask]).mean())

                ou_by_date = (
                    scored_history.groupby("_eval_date")
                    .apply(_ou_acc)
                    .reset_index()
                    .rename(columns={"_eval_date": "date", 0: "O/U Accuracy"})
                    .sort_values("date")
                )
                ou_by_date["date"] = ou_by_date["date"].astype(str)
                st.bar_chart(ou_by_date.set_index("date")["O/U Accuracy"])
            else:
                st.caption(
                    "O/U accuracy requires `predicted_strikeouts` and "
                    "`actual_strikeouts` columns."
                )

        # Summary stats
        st.subheader("30-Day Summary Metrics")
        sum_cols = st.columns(4)
        if "abs_error" in scored_history.columns:
            mae_val = scored_history["abs_error"].mean()
            sum_cols[0].metric("MAE", f"{mae_val:.3f}")
        if "signed_error" in scored_history.columns:
            rmse_val = (scored_history["signed_error"] ** 2).mean() ** 0.5
            bias_val = scored_history["signed_error"].mean()
            sum_cols[1].metric("RMSE", f"{rmse_val:.3f}")
            sum_cols[2].metric("Bias", f"{bias_val:+.3f}")
        sum_cols[3].metric("Samples", len(scored_history))

except Exception as exc:
    st.error(f"Error loading performance data: {exc}")

st.divider()

# ── Section 3: Feature Importance ───────────────────────────────────────────
st.header("Feature Importance (Top 15)")

try:
    feat_df = _load_feature_importance()
    if feat_df.empty:
        st.info(
            "No feature importance data found.  "
            f"Expected a CSV in `{PRODUCTION_MODEL_DIR}` named "
            "`shap_importance.csv`, `feature_importance.csv`, or `gain_importance.csv`."
        )
    else:
        # Detect the importance column
        importance_col = None
        for candidate in ("shap_importance", "gain_importance", "importance", "value"):
            if candidate in feat_df.columns:
                importance_col = candidate
                break
        feature_col = "feature" if "feature" in feat_df.columns else feat_df.columns[0]

        if importance_col:
            top15 = (
                feat_df[[feature_col, importance_col]]
                .sort_values(importance_col, ascending=False)
                .head(15)
                .sort_values(importance_col, ascending=True)  # ascending for horizontal feel
                .set_index(feature_col)
            )
            st.bar_chart(top15[importance_col])
            with st.expander("View full importance table"):
                st.dataframe(
                    feat_df[[feature_col, importance_col]]
                    .sort_values(importance_col, ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True,
                )
        else:
            st.dataframe(feat_df, use_container_width=True)
except Exception as exc:
    st.error(f"Error loading feature importance: {exc}")

st.divider()

# ── Section 4: Prediction History per Pitcher ───────────────────────────────
st.header("Prediction History")

try:
    # Collect pitcher IDs from daily predictions (if loaded) or scored history
    pitcher_options: list[str] = []
    if daily_preds is not None and not daily_preds.empty and "pitcher_id" in daily_preds.columns:
        pitcher_options = sorted(daily_preds["pitcher_id"].astype(str).unique().tolist())

    scored_history_local = _load_scored_history(days=30)
    if not scored_history_local.empty and "pitcher_id" in scored_history_local.columns:
        extra = sorted(scored_history_local["pitcher_id"].astype(str).unique().tolist())
        pitcher_options = sorted(set(pitcher_options + extra))

    if not pitcher_options:
        st.info("No pitcher data available to build the history dropdown.")
    else:
        selected_pitcher = st.selectbox(
            "Select Pitcher ID",
            options=pitcher_options,
            index=0,
        )

        hist_col1, hist_col2 = st.columns([3, 2])

        with hist_col1:
            st.subheader(f"Last 10 Predictions — Pitcher {selected_pitcher}")
            preds_for_pitcher = _load_all_predictions_for_pitcher(selected_pitcher, n=10)
            if preds_for_pitcher.empty:
                st.caption("No prediction history found.")
            else:
                show_cols = [c for c in [
                    "game_date", "game_pk",
                    "predicted_strikeouts", "pred_ci_lower", "pred_ci_upper",
                ] if c in preds_for_pitcher.columns]
                st.dataframe(
                    preds_for_pitcher[show_cols].reset_index(drop=True),
                    use_container_width=True,
                )

        with hist_col2:
            st.subheader("Actual vs Predicted")
            actuals = _load_actuals_for_pitcher(selected_pitcher)
            if actuals.empty:
                st.caption("No actuals recorded yet.")
            elif not preds_for_pitcher.empty:
                # Merge predictions + actuals on date if possible
                if (
                    "game_date" in preds_for_pitcher.columns
                    and "actual_strikeouts" in actuals.columns
                    and "_eval_date" in actuals.columns
                ):
                    merged = preds_for_pitcher[["game_date", "predicted_strikeouts"]].merge(
                        actuals[["_eval_date", "actual_strikeouts"]].rename(
                            columns={"_eval_date": "game_date"}
                        ),
                        on="game_date",
                        how="inner",
                    ).sort_values("game_date")
                    if not merged.empty:
                        st.line_chart(
                            merged.set_index("game_date")[
                                ["predicted_strikeouts", "actual_strikeouts"]
                            ]
                        )
                    else:
                        st.caption("No overlapping dates between predictions and actuals.")
                else:
                    st.dataframe(actuals.head(10), use_container_width=True)
            else:
                st.dataframe(actuals.head(10), use_container_width=True)

except Exception as exc:
    st.error(f"Error loading prediction history: {exc}")
