"""
validate_history.py — Grade the ACTUAL posted Discord picks (transcribed below)
against real strikeouts from Statcast, rebuild data/predictions/results_log.csv for
those dates, and post a consolidated "week in review" card to Discord.

Run: python tools/validate_history.py            # grade + update log + post
     python tools/validate_history.py --dry-run   # don't post
"""
from __future__ import annotations
import argparse, sys, unicodedata, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
from pybaseball import statcast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.daily_picks import load_env, webhook_url, send_discord
from tools.results import update_log, rolling_stats, EMOJI, JUICE_WIN, K_EVENTS, LOG

# Ground truth: exactly what was posted to Discord each morning (pitcher, OVER line).
POSTED = {
    "2026-06-23": [("Sonny Gray", 2.5), ("Carlos Rodón", 2.0), ("Jesús Luzardo", 2.0),
                   ("Shane Baz", 2.0), ("George Kirby", 2.0)],
    "2026-06-24": [("Tarik Skubal", 3.0), ("Ranger Suarez", 2.5), ("Braxton Ashcraft", 2.5),
                   ("Jacob deGrom", 2.0), ("Joe Ryan", 2.0)],
    "2026-06-25": [("Tatsuya Imai", 2.5), ("Cam Schlittler", 2.5), ("Cristopher Sánchez", 2.5),
                   ("Bryce Miller", 2.0), ("Freddy Peralta", 2.0)],
    "2026-06-26": [("Paul Skenes", 3.5), ("Jacob Misiorowski", 2.5), ("Zack Wheeler", 2.5),
                   ("Joey Cantillo", 2.0), ("Taj Bradley", 2.0)],
    "2026-06-27": [("Dylan Cease", 3.0), ("Chase Burns", 3.0), ("Jack Perkins", 2.5),
                   ("Logan Gilbert", 2.5), ("Jared Jones", 2.5)],
    "2026-06-28": [("Chris Sale", 3.0), ("Jesús Luzardo", 2.5), ("Hunter Brown", 2.5),
                   ("Carlos Rodón", 2.5), ("Gavin Williams", 2.5)],
    "2026-06-29": [("George Kirby", 2.5), ("Aaron Nola", 2.5), ("Braxton Ashcraft", 2.0),
                   ("Gage Jump", 2.0), ("Shane Baz", 2.0)],
}


def deaccent(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def name_key(s: str) -> str:
    s = deaccent(s).strip()
    if "," in s:                       # "Rodon, Carlos" -> "carlos rodon"
        last, first = [x.strip() for x in s.split(",", 1)]
        s = f"{first} {last}"
    return " ".join(s.lower().replace(".", "").split())


def actuals_for_date(date_str: str):
    """name_key -> strikeouts that day; plus set of who pitched (for void detection)."""
    df = statcast(start_dt=date_str, end_dt=date_str)
    if df is None or df.empty:
        return {}, set()
    appeared = {name_key(n): n for n in df["player_name"].dropna().unique()}
    ks = (df[df["events"].isin(K_EVENTS)]
          .groupby("player_name").size())
    kmap = {}
    for n, c in ks.items():
        kmap[name_key(n)] = int(c)
    return kmap, set(appeared.keys())


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhook")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    url = webhook_url(args.webhook)

    all_graded, day_records = [], []
    for date_str, picks in POSTED.items():
        kmap, appeared = actuals_for_date(date_str)
        graded = []
        for name, line in picks:
            k = name_key(name)
            if k in kmap:
                actual = kmap[k]
            elif k in appeared:
                actual = 0                 # pitched but no Ks
            else:
                actual = None              # didn't pitch / no match
            if actual is None:
                res = "VOID"
            elif actual == line:
                res = "PUSH"
            elif actual > line:
                res = "WIN"
            else:
                res = "LOSS"
            g = {"pitcher": name, "pitcher_id": None, "line": line, "lean": "OVER",
                 "actual_K": actual, "result": res}
            graded.append(g)
        update_log(date_str, graded)       # idempotent per date
        all_graded += [(date_str, g) for g in graded]
        w = sum(g["result"] == "WIN" for g in graded)
        l = sum(g["result"] == "LOSS" for g in graded)
        day_records.append((date_str, w, l, graded))

    # console + Discord summary
    log = pd.read_csv(LOG)
    tot_w = sum(r[1] for r in day_records)
    tot_l = sum(r[2] for r in day_records)
    decided = tot_w + tot_l
    units = tot_w * JUICE_WIN - tot_l
    hr = tot_w / decided * 100 if decided else 0

    lines_out = [f"**Validated {len(all_graded)} posted calls (6/23–6/29)**",
                 f"Record: **{tot_w}-{tot_l}** ({hr:.0f}%) · {units:+.1f}u\n"]
    for date_str, w, l, graded in day_records:
        tag = ", ".join(f"{EMOJI[g['result']]}{g['pitcher'].split()[-1]} "
                        f"{g['actual_K'] if g['actual_K'] is not None else 'DNP'}" for g in graded)
        lines_out.append(f"`{date_str[5:]}` {w}-{l}: {tag}")
    body = "\n".join(lines_out)

    print(body)
    for date_str, g in all_graded:
        print(f"  {date_str} {EMOJI[g['result']]} {g['pitcher']} O{g['line']} -> {g['actual_K']}")

    if not args.dry_run:
        send_discord(url, "📋 Backlog validated — full record", body, color=0x9b59b6)
    print(f"\nLog now spans {log['date'].nunique()} days, {len(log)} graded picks.")


if __name__ == "__main__":
    main()
