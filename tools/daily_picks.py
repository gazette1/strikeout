"""
daily_picks.py — Generate today's strikeout projections from the trained model and
send them to Discord.

Pipeline:
  1. Refresh the current-month Statcast cache (so pitcher state is current).
  2. Compute each pitcher's current "as-of" state and each team's batting K% (overall + vs hand).
  3. Pull the day's schedule + probable starters from the MLB Stats API.
  4. Predict median K + 90% interval for each probable starter with the saved quantile models.
  5. Rank, optionally filter by edge vs supplied lines, and post a Discord embed.

Setup: put your webhook in .env  ->  DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
Run:   python tools/daily_picks.py                 # today
       python tools/daily_picks.py --date 2026-06-24
       python tools/daily_picks.py --lines lines.json --min-edge 1.0
       python tools/daily_picks.py --test           # send a test message only
       python tools/daily_picks.py --dry-run        # print, don't send
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from datetime import date as date_cls
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import requests
import statsapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.build_dataset import (load_pitches, identify_starts, aggregate_starts,
                                 K_EVENTS, CACHE)
from tools.backfill_statcast import pull_month

MODELS = ROOT / "data" / "models" / "production"
FEATURES = ["k_pct_career", "swstr_pct", "csw_pct", "putaway_rate", "bf_per_start",
            "k_last3", "fb_velo_asof", "days_rest", "opp_k_pct", "opp_k_pct_vs_hand",
            "is_home", "n_prior_starts"]


# ── env / .env ────────────────────────────────────────────────────────────
def load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def webhook_url(arg):
    return arg or os.getenv("DISCORD_WEBHOOK_URL")


# ── Discord ───────────────────────────────────────────────────────────────
def send_discord(url, title, body, color=0x1abc9c):
    if not url:
        print("[no webhook] would have sent:\n" + body)
        return False
    payload = {"embeds": [{"title": title, "description": body[:4000], "color": color}]}
    try:
        r = requests.post(url, json=payload, timeout=15)
        ok = 200 <= r.status_code < 300
        print(f"Discord: HTTP {r.status_code} ({'sent' if ok else 'FAILED'})")
        return ok
    except Exception as e:
        print(f"Discord send failed: {e}")
        return False


# ── model ─────────────────────────────────────────────────────────────────
def load_models():
    out = {}
    for name in ["lower", "median", "upper"]:
        p = MODELS / f"lgbm_{name}.txt"
        if not p.exists():
            raise SystemExit(f"Missing model {p}. Run tools/train_model.py first.")
        out[name] = lgb.Booster(model_file=str(p))
    return out


def predict(models, X):
    lo, md, hi = (models[k].predict(X) for k in ["lower", "median", "upper"])
    md = np.clip(md, 0, 20)
    lo = np.clip(np.minimum(lo, md), 0, 20)
    hi = np.clip(np.maximum(hi, md), 0, 20)
    return lo, md, hi


# ── current state from pitch cache ────────────────────────────────────────
def pitcher_state(pitches):
    s = aggregate_starts(identify_starts(pitches))
    g = s.sort_values("game_date").groupby("pitcher")
    last3 = g.tail(3).groupby("pitcher").agg(k3=("K", "sum"), bf3=("bf", "sum"))
    st = g.agg(
        player_name=("player_name", "first"),
        cum_K=("K", "sum"), cum_bf=("bf", "sum"), cum_pitches=("pitches", "sum"),
        cum_swstr=("swstr", "sum"), cum_csw=("csw", "sum"), cum_twok=("two_k_pa", "sum"),
        fb=("fb_velo", "mean"), nstarts=("K", "size"), last_date=("game_date", "max"),
    )
    st = st.join(last3)
    st["k_pct_career"] = st.cum_K / st.cum_bf
    st["swstr_pct"] = st.cum_swstr / st.cum_pitches
    st["csw_pct"] = st.cum_csw / st.cum_pitches
    st["putaway_rate"] = st.cum_K / st.cum_twok.replace(0, np.nan)
    st["bf_per_start"] = st.cum_bf / st.nstarts
    st["k_last3"] = st.k3 / st.bf3
    st["fb_velo_asof"] = st.fb
    st["n_prior_starts"] = st.nstarts
    return st


def team_state(pitches):
    p = pitches.dropna(subset=["events"]).copy()
    p["bat_team"] = np.where(p["inning_topbot"] == "Top", p["away_team"], p["home_team"])
    p["is_k"] = p["events"].isin(K_EVENTS)
    p["pa"] = p["game_pk"].astype(str) + "_" + p["at_bat_number"].astype(str)
    overall = p.groupby("bat_team").agg(K=("is_k", "sum"), PA=("pa", "nunique"))
    overall["opp_k_pct"] = overall.K / overall.PA
    vs = p.groupby(["bat_team", "p_throws"]).agg(K=("is_k", "sum"), PA=("pa", "nunique"))
    vs["opp_k_pct_vs_hand"] = vs.K / vs.PA
    return overall["opp_k_pct"], vs["opp_k_pct_vs_hand"]


def norm(s):
    s = str(s).strip()
    if "," in s:
        last, first = [x.strip() for x in s.split(",", 1)]
        s = f"{first} {last}"
    return " ".join(s.lower().replace(".", "").split())


# ── main ──────────────────────────────────────────────────────────────────
def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date_cls.today()))
    ap.add_argument("--lines", help="JSON file mapping pitcher name -> K line")
    ap.add_argument("--min-edge", type=float, default=1.0,
                    help="only flag picks whose |proj-line| exceeds this (needs --lines)")
    ap.add_argument("--top", type=int, default=5, help="number of conviction plays to send")
    ap.add_argument("--webhook", help="Discord webhook URL (else DISCORD_WEBHOOK_URL)")
    ap.add_argument("--no-refresh", action="store_true", help="skip Statcast refresh")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    url = webhook_url(args.webhook)

    if args.test:
        ok = send_discord(url, "✅ StrikeOut Bot connected",
                          "Discord alerts are wired up. You'll get daily K projections here.")
        sys.exit(0 if ok else 1)

    # 1) refresh current month so pitcher state is current
    if not args.no_refresh:
        y, m = int(args.date[:4]), int(args.date[5:7])
        f = CACHE / f"{y}-{m:02d}.parquet"
        if f.exists():
            f.unlink()  # force re-pull for freshness
        pull_month(y, m)

    # 2) state
    pitches = load_pitches()
    pstate = pitcher_state(pitches)
    opp_k, opp_k_hand = team_state(pitches)
    pstate["name_key"] = pstate["player_name"].map(norm)
    name_to_idx = pstate.reset_index().set_index("name_key")

    # team id -> abbrev
    teams = statsapi.get("teams", {"sportId": 1})["teams"]
    id2abbr = {t["id"]: t["abbreviation"] for t in teams}

    # 3) schedule + probables
    sched = statsapi.schedule(start_date=args.date, end_date=args.date)
    lines = {}
    if args.lines and Path(args.lines).exists():
        lines = {norm(k): v for k, v in json.loads(Path(args.lines).read_text()).items()}

    models = load_models()
    rows = []
    for g in sched:
        home, away = id2abbr.get(g["home_id"]), id2abbr.get(g["away_id"])
        for who, opp, is_home in [("home_probable_pitcher", away, 1),
                                  ("away_probable_pitcher", home, 0)]:
            pname = g.get(who)
            if not pname:
                continue
            key = norm(pname)
            if key not in name_to_idx.index:
                rows.append({"pitcher": pname, "note": "no history", "proj": None})
                continue
            ps = name_to_idx.loc[key]
            if isinstance(ps, pd.DataFrame):
                ps = ps.iloc[0]
            hand = pitches.loc[pitches["pitcher"] == ps["pitcher"], "p_throws"]
            hand = hand.iloc[-1] if len(hand) else "R"
            days_rest = (pd.Timestamp(args.date) - ps["last_date"]).days
            feat = {
                "k_pct_career": ps["k_pct_career"], "swstr_pct": ps["swstr_pct"],
                "csw_pct": ps["csw_pct"], "putaway_rate": ps["putaway_rate"],
                "bf_per_start": ps["bf_per_start"], "k_last3": ps["k_last3"],
                "fb_velo_asof": ps["fb_velo_asof"],
                "days_rest": min(max(days_rest, 0), 30),
                "opp_k_pct": opp_k.get(opp, np.nan),
                "opp_k_pct_vs_hand": opp_k_hand.get((opp, hand), opp_k.get(opp, np.nan)),
                "is_home": is_home, "n_prior_starts": ps["n_prior_starts"],
            }
            X = pd.DataFrame([feat])[FEATURES]
            lo, md, hi = predict(models, X)
            r = {"pitcher": pname, "opp": opp, "hand": hand,
                 "proj": round(float(md[0]), 1), "lo": round(float(lo[0]), 1),
                 "hi": round(float(hi[0]), 1)}
            if key in lines:
                r["line"] = lines[key]
                r["edge"] = round(r["proj"] - lines[key], 1)
                r["lean"] = "OVER" if r["edge"] > 0 else "UNDER"
            rows.append(r)

    have = [r for r in rows if r.get("proj") is not None]
    # conviction = model's 5th-percentile floor: highest "confident OVER" line.
    # suggested line = floor rounded down to nearest 0.5.
    for r in have:
        r["conf_line"] = np.floor(r["lo"] * 2) / 2
        r["conviction"] = round(r["lo"], 2)  # rank key
    have.sort(key=lambda r: (r["conviction"], r["proj"]), reverse=True)

    def stars(r):
        return "🔒" if r["conf_line"] >= 4.5 else "⭐" if r["conf_line"] >= 3.5 else "•"

    # 4) format
    if not have:
        body = f"No probable starters with history found for {args.date}."
    elif lines:
        flagged = [r for r in have if abs(r.get("edge", 0)) >= args.min_edge]
        flagged.sort(key=lambda r: abs(r.get("edge", 0)), reverse=True)
        lines_out = [f"`{r['lean']:5s} {r['line']:>4}` **{r['pitcher']}** vs {r['opp']} "
                     f"-> proj {r['proj']} [{r['lo']}-{r['hi']}] (edge {r['edge']:+})"
                     for r in flagged]
        body = ("\n".join(lines_out) if lines_out
                else f"No plays cleared |edge| >= {args.min_edge} today.")
    else:
        top = have[:args.top]
        lines_out = [
            f"{stars(r)} **{r['pitcher']}** ({r['hand']}) vs {r['opp']}\n"
            f" OVER **{r['conf_line']:.1f}** K  ·  proj {r['proj']}  ·  range [{r['lo']}–{r['hi']}]"
            for r in top]
        body = ("**Top conviction strikeout plays** — line = model's ~95% floor\n\n"
                + "\n".join(lines_out))
        miss = [r["pitcher"] for r in rows if r.get("proj") is None]
        if miss:
            body += f"\n\n_No history (skipped): {', '.join(miss[:8])}_"

    title = f"⚾ Top K Plays — {args.date}"
    print(title + "\n" + body)
    if not have:
        print("No actionable plays (off-day or probables not posted) — skipping Discord send.")
        return
    if not args.dry_run:
        send_discord(url, title, body)


if __name__ == "__main__":
    main()
