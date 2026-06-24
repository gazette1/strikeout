"""
train_model.py — Train LightGBM quantile models on the real Statcast-derived dataset,
validate honestly with walk-forward CV, and compare to the naive baseline.

- Quantile models at 0.05 / 0.50 / 0.95 -> median prediction + 90% interval.
- Walk-forward: for each 2025 month, train on everything strictly before it, test on it.
- Baseline: expected_K = career K% * prior batters-faced-per-start (no leakage).
- Saves production models + metadata to data/models/production/.

Run: python tools/train_model.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "features" / "training.parquet"
MODELS = ROOT / "data" / "models" / "production"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES = ["k_pct_career", "swstr_pct", "csw_pct", "putaway_rate", "bf_per_start",
            "k_last3", "fb_velo_asof", "days_rest", "opp_k_pct", "opp_k_pct_vs_hand",
            "is_home", "n_prior_starts"]
TARGET = "K"
LINES = [3.5, 4.5, 5.5, 6.5, 7.5]

BASE = dict(objective="quantile", num_leaves=31, learning_rate=0.05, n_estimators=400,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=30, reg_lambda=1.0, verbose=-1, seed=42)


def fit_quantiles(Xtr, ytr):
    models = {}
    # wider tails (0.03/0.97) so the empirical 90% interval isn't under-covered
    for name, a in [("lower", 0.03), ("median", 0.5), ("upper", 0.97)]:
        m = lgb.LGBMRegressor(**{**BASE, "alpha": a})
        m.fit(Xtr, ytr)
        models[name] = m
    return models


def predict(models, X):
    lo = models["lower"].predict(X)
    md = models["median"].predict(X)
    hi = models["upper"].predict(X)
    md = np.clip(md, 0, 20)
    lo = np.clip(np.minimum(lo, md), 0, 20)
    hi = np.clip(np.maximum(hi, md), 0, 20)
    return lo, md, hi


def metrics(y, pred, lo=None, hi=None, baseline=None):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {"n": int(len(y)), "MAE": round(float(np.mean(np.abs(y - pred))), 3),
           "RMSE": round(float(np.sqrt(np.mean((y - pred) ** 2))), 3)}
    if lo is not None:
        out["cov90"] = round(float(np.mean((y >= lo) & (y <= hi))) * 100, 1)
    if baseline is not None:
        out["baseline_MAE"] = round(float(np.mean(np.abs(y - baseline))), 3)
    # over/under accuracy vs lines (skip pushes)
    accs = []
    for ln in LINES:
        mask = y != ln
        if mask.sum():
            accs.append(np.mean((pred[mask] > ln) == (y[mask] > ln)))
    out["OU_acc"] = round(float(np.mean(accs)) * 100, 1)
    return out


def main():
    df = pd.read_parquet(DATA).sort_values("game_date").reset_index(drop=True)
    df["month"] = df["game_date"].dt.to_period("M").astype(str)
    df["baseline"] = df["k_pct_career"] * df["bf_per_start"]

    # ---- Walk-forward over 2025 months ----
    test_months = [m for m in sorted(df["month"].unique()) if m.startswith("2025")]
    rows, all_pred = [], []
    print("Walk-forward CV (train on all prior dates, test on each 2025 month):")
    for m in test_months:
        tr = df[df["month"] < m]
        te = df[df["month"] == m]
        if len(tr) < 500 or te.empty:
            continue
        models = fit_quantiles(tr[FEATURES], tr[TARGET])
        lo, md, hi = predict(models, te[FEATURES])
        mt = metrics(te[TARGET], md, lo, hi, te["baseline"])
        mt["month"] = m
        rows.append(mt)
        tmp = te[["game_date", "player_name", "opp_team", TARGET, "baseline"]].copy()
        tmp["pred"], tmp["lo"], tmp["hi"] = md, lo, hi
        all_pred.append(tmp)
        print(f"  {m}: n={mt['n']:4d}  MAE={mt['MAE']:.3f} (base {mt['baseline_MAE']:.3f})  "
              f"cov90={mt['cov90']:.1f}%  OU={mt['OU_acc']:.1f}%")

    wf = pd.concat(all_pred, ignore_index=True)
    agg = metrics(wf[TARGET], wf["pred"], wf["lo"], wf["hi"], wf["baseline"])
    print("\n== POOLED WALK-FORWARD ==")
    print(f"  Model    MAE {agg['MAE']:.3f} | RMSE {agg['RMSE']:.3f} | cov90 {agg['cov90']:.1f}% "
          f"| O/U {agg['OU_acc']:.1f}%  (n={agg['n']})")
    print(f"  Baseline MAE {agg['baseline_MAE']:.3f}")
    lift = (agg['baseline_MAE'] - agg['MAE']) / agg['baseline_MAE'] * 100
    print(f"  -> Model beats baseline by {lift:+.1f}% MAE")

    # ---- Fit final production model on ALL data ----
    final = fit_quantiles(df[FEATURES], df[TARGET])
    for name, m in final.items():
        m.booster_.save_model(str(MODELS / f"lgbm_{name}.txt"))
    meta = {"features": FEATURES, "target": TARGET,
            "trained_rows": int(len(df)),
            "date_range": [str(df["game_date"].min().date()), str(df["game_date"].max().date())],
            "walk_forward": agg, "per_month": rows}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))

    # feature importance (gain)
    imp = pd.Series(final["median"].booster_.feature_importance("gain"),
                    index=FEATURES).sort_values(ascending=False)
    print("\nTop features (gain):")
    print((imp / imp.sum() * 100).round(1).to_string())
    print(f"\nSaved production models + metadata -> {MODELS}")


if __name__ == "__main__":
    main()
