"""
eval_vs_log.py — Does the real (Statcast-trained) model make better calls than the bot?

Trains quantile models on PRE-2026 data only, predicts each pitcher-start in 2026,
matches them to the actual betting log (`Baseball Log .xlsx`) by pitcher + date, and
compares the model's OVER/UNDER calls to the bot's logged calls on the same games.

Run: python tools/eval_vs_log.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "features" / "training.parquet"
LOG = ROOT / "Baseball Log .xlsx"
JUICE_WIN = 100 / 110

FEATURES = ["k_pct_career", "swstr_pct", "csw_pct", "putaway_rate", "bf_per_start",
            "k_last3", "fb_velo_asof", "days_rest", "opp_k_pct", "opp_k_pct_vs_hand",
            "is_home", "n_prior_starts"]
BASE = dict(objective="quantile", alpha=0.5, num_leaves=31, learning_rate=0.05,
            n_estimators=400, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, min_child_samples=30, reg_lambda=1.0, verbose=-1, seed=42)


def norm_name(s: str) -> str:
    s = str(s).strip()
    if "," in s:                       # "Rodriguez, Eduardo" -> "eduardo rodriguez"
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}"
    return " ".join(s.lower().replace(".", "").split())


def grade(call, actual, line):
    if actual == line:
        return None
    over = actual > line
    return float(call == ("OVER" if over else "UNDER"))


def main():
    df = pd.read_parquet(DATA)
    df["game_date"] = pd.to_datetime(df["game_date"])
    train = df[df["game_date"] < "2026-01-01"]
    test = df[df["game_date"] >= "2026-01-01"].copy()
    print(f"Train rows (pre-2026): {len(train):,} | 2026 starts available: {len(test):,}")
    if test.empty:
        raise SystemExit("No 2026 rows yet — run backfill_statcast.py 2026 then build_dataset.py")

    model = lgb.LGBMRegressor(**BASE).fit(train[FEATURES], train["K"])
    test["model_pred"] = np.clip(model.predict(test[FEATURES]), 0, 20)
    test["name_key"] = test["player_name"].map(norm_name)
    test["date_key"] = test["game_date"].dt.date.astype(str)

    log = pd.read_excel(LOG, sheet_name="Sheet1")
    log.columns = [c.strip() for c in log.columns]
    log = log[log["Result"].notna()].copy()
    log["name_key"] = log["Pitcher"].map(norm_name)
    log["date_key"] = pd.to_datetime(log["Date"]).dt.date.astype(str)
    for c in ["K Line", "Actual K"]:
        log[c] = pd.to_numeric(log[c], errors="coerce")
    log["bot_call"] = log["Call"].astype(str).str.upper().str.strip()
    log["bot_hit"] = (log["Result"].astype(str).str.upper() == "HIT").astype(int)

    m = log.merge(test[["name_key", "date_key", "model_pred", "K"]],
                  on=["name_key", "date_key"], how="inner")
    print(f"Matched {len(m)} of {len(log)} graded log picks to model predictions\n")
    if m.empty:
        raise SystemExit("No name/date matches — check 2026 backfill coverage.")

    # data sanity: logged Actual K vs Statcast-derived K
    sane = (m["Actual K"] == m["K"]).mean() * 100
    print(f"Sanity: logged Actual K matches Statcast K on {sane:.0f}% of matches\n")

    m["model_call"] = np.where(m["model_pred"] > m["K Line"], "OVER", "UNDER")
    m["edge"] = (m["model_pred"] - m["K Line"]).abs()
    m["model_won"] = [grade(c, a, l) for c, a, l in zip(m["model_call"], m["Actual K"], m["K Line"])]
    m["bot_won"] = [grade(c, a, l) for c, a, l in zip(m["bot_call"], m["Actual K"], m["K Line"])]

    def summ(mask, label):
        d = m[mask]
        for who in ["bot", "model"]:
            w = d[f"{who}_won"].dropna()
            if len(w):
                u = sum(JUICE_WIN if x == 1 else -1 for x in w)
                print(f"  {label:22s} {who:5s}: {w.mean()*100:5.1f}% hit, "
                      f"{u/len(w):+.3f} u/bet  (n={len(w)})")

    print("HEAD-TO-HEAD on the same games:")
    summ(m.index >= 0, "ALL matched")
    for t in [0.5, 1.0, 1.5]:
        summ(m["edge"] > t, f"model |edge|>{t}")
    print(f"\n  Agreement (same call): {(m['model_call']==m['bot_call']).mean()*100:.0f}%")
    print(f"  Model OVER rate {(m['model_call']=='OVER').mean()*100:.0f}% vs "
          f"bot OVER rate {(m['bot_call']=='OVER').mean()*100:.0f}%")


if __name__ == "__main__":
    main()
