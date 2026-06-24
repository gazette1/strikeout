"""
build_dataset.py — Turn cached Statcast pitches into a leakage-safe training table.

One row per starting-pitcher start. Target = strikeouts recorded by that starter.
All features are computed from games STRICTLY BEFORE the start date (no leakage).

Output: data/features/training.parquet

Run: python tools/build_dataset.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "statcast"
OUT = ROOT / "data" / "features"
OUT.mkdir(parents=True, exist_ok=True)

K_EVENTS = {"strikeout", "strikeout_double_play"}
SWSTR = {"swinging_strike", "swinging_strike_blocked"}
CSW = SWSTR | {"called_strike"}
FB = {"FF", "SI", "FC"}


def load_pitches() -> pd.DataFrame:
    files = sorted(CACHE.glob("*.parquet"))
    if not files:
        raise SystemExit("No cached Statcast months. Run tools/backfill_statcast.py first.")
    cols = ["game_pk", "game_date", "pitcher", "player_name", "batter", "p_throws",
            "home_team", "away_team", "inning_topbot", "at_bat_number",
            "events", "description", "pitch_type", "release_speed", "strikes"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"Loaded {len(df):,} pitches from {len(files)} months "
          f"({df['game_date'].min().date()} -> {df['game_date'].max().date()})")
    return df


def identify_starts(df: pd.DataFrame) -> pd.DataFrame:
    """For each (game, half-inning side), the starter = earliest at_bat_number pitcher.
    Top half => away batting => HOME pitcher; Bot half => AWAY pitcher."""
    # starter per game per side
    side = df.dropna(subset=["pitcher", "at_bat_number"]).copy()
    first = (side.sort_values("at_bat_number")
                 .groupby(["game_pk", "inning_topbot"])["pitcher"].first()
                 .rename("starter").reset_index())
    df = df.merge(first, on=["game_pk", "inning_topbot"], how="left")
    starter_rows = df[df["pitcher"] == df["starter"]].copy()
    starter_rows["is_home"] = (starter_rows["inning_topbot"] == "Top").astype(int)
    starter_rows["pitch_team"] = np.where(starter_rows["is_home"] == 1,
                                          starter_rows["home_team"], starter_rows["away_team"])
    starter_rows["opp_team"] = np.where(starter_rows["is_home"] == 1,
                                        starter_rows["away_team"], starter_rows["home_team"])
    return starter_rows


def aggregate_starts(s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby(["game_pk", "pitcher"])
    out = pd.DataFrame({
        "game_date": g["game_date"].first(),
        "player_name": g["player_name"].first(),
        "p_throws": g["p_throws"].first(),
        "is_home": g["is_home"].first(),
        "opp_team": g["opp_team"].first(),
        "pitch_team": g["pitch_team"].first(),
        "pitches": g.size(),
        "bf": g["at_bat_number"].nunique(),
        "K": g["events"].apply(lambda e: e.isin(K_EVENTS).sum()),
        "swstr": g["description"].apply(lambda d: d.isin(SWSTR).sum()),
        "csw": g["description"].apply(lambda d: d.isin(CSW).sum()),
        "fb_velo": s[s["pitch_type"].isin(FB)].groupby(["game_pk", "pitcher"])["release_speed"].mean(),
        # two-strike PAs: at-bats that reached a 2-strike count (for putaway rate)
        "two_k_pa": s[s["strikes"] == 2].groupby(["game_pk", "pitcher"])["at_bat_number"].nunique(),
    }).reset_index()
    out["two_k_pa"] = out["two_k_pa"].fillna(0)
    # keep only real starts (filters relievers caught by the heuristic)
    out = out[out["bf"] >= 8].sort_values("game_date").reset_index(drop=True)
    return out


def team_k_pct_asof(s: pd.DataFrame, starts: pd.DataFrame) -> pd.Series:
    """Opponent batting K% from all their games strictly before each start date."""
    # per game, per batting team: K and PA
    bat = s.copy()
    bat["bat_team"] = bat["opp_team"]  # opp of the pitcher = batting team this half
    tg = (bat.groupby(["game_pk", "bat_team", "game_date"])
             .agg(K=("events", lambda e: e.isin(K_EVENTS).sum()),
                  PA=("at_bat_number", "nunique")).reset_index()
             .sort_values("game_date"))
    tg["cumK"] = tg.groupby("bat_team")["K"].cumsum() - tg["K"]
    tg["cumPA"] = tg.groupby("bat_team")["PA"].cumsum() - tg["PA"]
    tg["opp_k_pct_asof"] = (tg["cumK"] / tg["cumPA"]).replace([np.inf, -np.inf], np.nan)
    key = tg.set_index(["game_pk", "bat_team"])["opp_k_pct_asof"]
    return starts.set_index(["game_pk", "opp_team"]).index.map(key)


def opp_k_pct_vs_hand_asof(s: pd.DataFrame, starts: pd.DataFrame) -> pd.Series:
    """Batting team's K% vs pitchers of THIS hand, from prior games only."""
    bat = s.copy()
    bat["bat_team"] = bat["opp_team"]
    tg = (bat.groupby(["game_pk", "bat_team", "p_throws", "game_date"])
             .agg(K=("events", lambda e: e.isin(K_EVENTS).sum()),
                  PA=("at_bat_number", "nunique")).reset_index()
             .sort_values("game_date"))
    grp = tg.groupby(["bat_team", "p_throws"])
    tg["cumK"] = grp["K"].cumsum() - tg["K"]
    tg["cumPA"] = grp["PA"].cumsum() - tg["PA"]
    tg["val"] = (tg["cumK"] / tg["cumPA"]).replace([np.inf, -np.inf], np.nan)
    key = tg.set_index(["game_pk", "bat_team", "p_throws"])["val"]
    idx = starts.set_index(["game_pk", "opp_team", "p_throws"]).index
    return idx.map(key)


def add_pitcher_rolling(starts: pd.DataFrame) -> pd.DataFrame:
    """Expanding (prior-only) pitcher features + recency. Uses transform to keep
    the index aligned with `df` throughout."""
    df = starts.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    gp = df.groupby("pitcher")

    def exp_sum(col):  # prior-only expanding sum
        return gp[col].transform(lambda x: x.shift().expanding().sum())

    def roll_sum(col, n):  # prior-only rolling sum over last n starts
        return gp[col].transform(lambda x: x.shift().rolling(n, min_periods=1).sum())

    for c in ["K", "bf", "pitches", "swstr", "csw", "two_k_pa"]:
        df[f"cum_{c}"] = exp_sum(c)
    df["k_pct_career"] = df["cum_K"] / df["cum_bf"]
    df["swstr_pct"] = df["cum_swstr"] / df["cum_pitches"]
    df["csw_pct"] = df["cum_csw"] / df["cum_pitches"]
    df["putaway_rate"] = (df["cum_K"] / df["cum_two_k_pa"]).replace([np.inf, -np.inf], np.nan)
    df["n_prior_starts"] = gp.cumcount()
    df["bf_per_start"] = df["cum_bf"] / df["n_prior_starts"].where(df["n_prior_starts"] > 0)
    # recent form: K% over last 3 prior starts
    df["k_last3"] = roll_sum("K", 3) / roll_sum("bf", 3)
    # fastball velo, prior-only expanding mean
    df["fb_velo_asof"] = gp["fb_velo"].transform(lambda x: x.shift().expanding().mean())
    # days rest
    df["days_rest"] = gp["game_date"].diff().dt.days.clip(0, 30)
    return df


def main():
    df = load_pitches()
    s = identify_starts(df)
    starts = aggregate_starts(s)
    print(f"Identified {len(starts):,} starter-starts")

    starts["opp_k_pct"] = team_k_pct_asof(s, starts).astype(float)
    starts["opp_k_pct_vs_hand"] = opp_k_pct_vs_hand_asof(s, starts).astype(float)
    starts = add_pitcher_rolling(starts)
    # fall back hand-specific opp K% to overall opp K% when sparse
    starts["opp_k_pct_vs_hand"] = starts["opp_k_pct_vs_hand"].fillna(starts["opp_k_pct"])

    feat_cols = ["k_pct_career", "swstr_pct", "csw_pct", "putaway_rate", "bf_per_start",
                 "k_last3", "fb_velo_asof", "days_rest", "opp_k_pct", "opp_k_pct_vs_hand",
                 "is_home", "n_prior_starts"]
    # require a minimum prior history so features are meaningful
    train = starts[starts["n_prior_starts"] >= 3].copy()
    train = train.dropna(subset=["k_pct_career", "swstr_pct", "opp_k_pct"])
    keep = ["game_pk", "game_date", "pitcher", "player_name", "opp_team",
            "p_throws", "bf", "K"] + feat_cols
    train = train[keep].reset_index(drop=True)
    out = OUT / "training.parquet"
    train.to_parquet(out, index=False)
    print(f"Wrote {len(train):,} training rows -> {out}")
    print(f"  date range {train['game_date'].min().date()} -> {train['game_date'].max().date()}")
    print(f"  target K: mean {train['K'].mean():.2f}, std {train['K'].std():.2f}, "
          f"max {train['K'].max()}")
    print("\nFeature null rates:")
    print((train[feat_cols].isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
