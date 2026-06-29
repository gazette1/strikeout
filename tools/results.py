"""
results.py — Grade a past day's posted picks against actual strikeouts and post a
results card to Discord. Maintains a rolling results log for self-tracking.

Reads data/predictions/picks_{date}.json (written by daily_picks.py), pulls actual
strikeouts from Statcast for that date, grades each pick (W/L/Push/Void), appends to
data/predictions/results_log.csv, and posts a Discord summary with the day's record
and a 30-day rolling hit rate.

Run: python tools/results.py                 # grades yesterday
     python tools/results.py --date 2026-06-27
     python tools/results.py --dry-run
"""
from __future__ import annotations
import argparse, json, sys, warnings
from datetime import date as date_cls, timedelta
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pybaseball import statcast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.daily_picks import load_env, webhook_url, send_discord

K_EVENTS = {"strikeout", "strikeout_double_play"}
PICKS_DIR = ROOT / "data" / "predictions"
LOG = PICKS_DIR / "results_log.csv"
JUICE_WIN = 100 / 110


def actual_ks(date_str: str) -> dict:
    """pitcher_id -> strikeouts recorded that date (from Statcast)."""
    df = statcast(start_dt=date_str, end_dt=date_str)
    if df is None or df.empty:
        return {}
    k = df[df["events"].isin(K_EVENTS)].groupby("pitcher").size()
    return {int(pid): int(n) for pid, n in k.items()}


def grade_pick(p: dict, ks: dict) -> dict:
    pid = p.get("pitcher_id")
    line = float(p["line"])
    lean = p.get("lean", "OVER").upper()
    actual = ks.get(int(pid)) if pid is not None else None
    if actual is None:
        res = "VOID"          # didn't pitch / postponed
    elif actual == line:
        res = "PUSH"
    elif (lean == "OVER" and actual > line) or (lean == "UNDER" and actual < line):
        res = "WIN"
    else:
        res = "LOSS"
    return {**p, "actual_K": actual, "result": res}


def update_log(date_str: str, graded: list) -> pd.DataFrame:
    rows = [{"date": date_str, "pitcher": g["pitcher"], "line": g["line"],
             "lean": g.get("lean", "OVER"), "actual_K": g["actual_K"],
             "result": g["result"]} for g in graded]
    new = pd.DataFrame(rows)
    if LOG.exists():
        old = pd.read_csv(LOG)
        old = old[old["date"] != date_str]          # idempotent re-grade
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LOG, index=False)
    return new


def rolling_stats(log: pd.DataFrame, days: int = 30) -> str:
    log = log.copy()
    log["date"] = pd.to_datetime(log["date"])
    cutoff = log["date"].max() - pd.Timedelta(days=days)
    recent = log[(log["date"] >= cutoff) & (log["result"].isin(["WIN", "LOSS"]))]
    if recent.empty:
        return ""
    w = (recent["result"] == "WIN").sum()
    l = (recent["result"] == "LOSS").sum()
    hr = w / (w + l) * 100
    units = w * JUICE_WIN - l
    return f"\n📈 **Last {days}d:** {w}-{l} ({hr:.0f}%) · {units:+.1f}u"


EMOJI = {"WIN": "✅", "LOSS": "❌", "PUSH": "➖", "VOID": "⬜"}


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date_cls.today() - timedelta(days=1)))
    ap.add_argument("--webhook")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    url = webhook_url(args.webhook)

    pf = PICKS_DIR / f"picks_{args.date}.json"
    if not pf.exists():
        print(f"No saved picks for {args.date} ({pf.name}) — nothing to grade.")
        return
    picks = json.loads(pf.read_text())["picks"]
    if not picks:
        print(f"No picks recorded for {args.date}.")
        return

    ks = actual_ks(args.date)
    graded = [grade_pick(p, ks) for p in picks]
    log = update_log(args.date, graded)

    wins = sum(g["result"] == "WIN" for g in graded)
    losses = sum(g["result"] == "LOSS" for g in graded)
    decided = wins + losses
    hr = (wins / decided * 100) if decided else 0
    units = wins * JUICE_WIN - losses

    head = (f"**{wins}-{losses}**" + (f" ({hr:.0f}%) · {units:+.1f}u" if decided else "")
            + "  on yesterday's plays")
    lines_out = []
    for g in sorted(graded, key=lambda x: (x["result"] != "WIN", -float(x["line"]))):
        a = g["actual_K"] if g["actual_K"] is not None else "DNP"
        lines_out.append(f"{EMOJI[g['result']]} **{g['pitcher']}** {g.get('lean','OVER')} "
                         f"{g['line']:.1f} -> {a} K")
    body = head + "\n\n" + "\n".join(lines_out) + rolling_stats(log)

    title = f"📊 Results — {args.date}"
    print(title + "\n" + body)
    if not args.dry_run:
        send_discord(url, title, body, color=0x3498db)


if __name__ == "__main__":
    main()
