"""
odds.py — Fetch MLB pitcher strikeout O/U lines from The Odds API.

Returns a {name_key -> {line, over, under, n_books, book}} map for a given date,
where name_key matches tools.daily_picks.norm() (deaccented "first last").

Costs ~1 credit per game (events list is free). Free tier = 500/mo.
Needs env ODDS_API_KEY.
"""
from __future__ import annotations
import os, sys, statistics, warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
warnings.filterwarnings("ignore")

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.daily_picks import norm

BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
ET = timezone(timedelta(hours=-4))   # EDT (summer season)


def _events(date_str, key):
    r = requests.get(f"{BASE}/events", params={"apiKey": key}, timeout=20)
    r.raise_for_status()
    out = []
    for e in r.json():
        ct = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if ct.astimezone(ET).date().isoformat() == date_str:
            out.append(e)
    return out


def fetch_strikeout_lines(date_str, key=None) -> dict:
    key = key or os.getenv("ODDS_API_KEY")
    if not key:
        print("[odds] no ODDS_API_KEY set — skipping live lines.")
        return {}
    try:
        events = _events(date_str, key)
    except Exception as e:
        print(f"[odds] events fetch failed: {e}")
        return {}
    print(f"[odds] {len(events)} games on {date_str}")

    # collect (point, over_price) and (point, under_price) pairs per pitcher across books
    agg = {}   # name_key -> {"name":, "overs":[(pt,price)], "unders":[(pt,price)], "books":set()}
    remaining = None
    for e in events:
        try:
            r = requests.get(f"{BASE}/events/{e['id']}/odds",
                             params={"apiKey": key, "regions": "us",
                                     "markets": "pitcher_strikeouts",
                                     "oddsFormat": "american"}, timeout=20)
            remaining = r.headers.get("x-requests-remaining", remaining)
            if r.status_code != 200:
                continue
            for bk in r.json().get("bookmakers", []):
                for m in bk.get("markets", []):
                    if m["key"] != "pitcher_strikeouts":
                        continue
                    for o in m["outcomes"]:
                        nm = o.get("description")
                        if not nm or o.get("point") is None:
                            continue
                        d = agg.setdefault(norm(nm), {"name": nm, "overs": [], "unders": [],
                                                      "books": set()})
                        d["books"].add(bk["key"])
                        if o["name"] == "Over":
                            d["overs"].append((o["point"], o["price"]))
                        elif o["name"] == "Under":
                            d["unders"].append((o["point"], o["price"]))
        except Exception:
            continue

    lines = {}
    for k, d in agg.items():
        if not d["overs"]:
            continue
        # main line = the quote whose Over price is closest to even (-110ish);
        # alt lines carry extreme prices, so this ignores them.
        point, over_price = min(d["overs"], key=lambda p: abs(p[1]))
        if not (2.0 <= point <= 9.5):       # implausible main K line -> skip as bad data
            continue
        under_price = next((pr for pt, pr in d["unders"] if pt == point), None)
        lines[k] = {"name": d["name"], "line": point, "over": over_price,
                    "under": under_price, "n_books": len(d["books"])}
    print(f"[odds] {len(lines)} pitchers with K lines | credits remaining: {remaining}")
    return lines


if __name__ == "__main__":
    import json
    d = sys.argv[1] if len(sys.argv) > 1 else datetime.now(ET).date().isoformat()
    print(json.dumps(fetch_strikeout_lines(d), indent=2)[:2000])
