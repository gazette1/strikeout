"""
backfill_statcast.py — Pull Statcast pitch-level data month-by-month and cache to parquet.

Statcast (Baseball Savant via pybaseball) is the one fully-free, reliable source and
contains everything we need to build training data: pitcher, batter, events, teams,
handedness, pitch characteristics. Each month is cached so re-runs are free.

Run: python tools/backfill_statcast.py 2024 2025
"""
from __future__ import annotations
import sys, time, warnings, calendar
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
from pybaseball import statcast

CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "statcast"
CACHE.mkdir(parents=True, exist_ok=True)

# MLB regular season roughly late March -> early October
SEASON_MONTHS = list(range(3, 11))  # Mar..Oct


def pull_month(year: int, month: int) -> int:
    out = CACHE / f"{year}-{month:02d}.parquet"
    if out.exists():
        n = len(pd.read_parquet(out, columns=["game_pk"]))
        print(f"  {year}-{month:02d}: cached ({n} rows)")
        return n
    last = calendar.monthrange(year, month)[1]
    start, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"
    try:
        df = statcast(start_dt=start, end_dt=end)
    except Exception as e:
        print(f"  {year}-{month:02d}: FAIL {repr(e)[:120]}")
        return 0
    if df is None or df.empty:
        print(f"  {year}-{month:02d}: 0 rows (off-season)")
        return 0
    df.to_parquet(out, index=False)
    print(f"  {year}-{month:02d}: pulled {len(df)} rows -> {out.name}")
    time.sleep(2)  # be polite
    return len(df)


def main():
    years = [int(a) for a in sys.argv[1:]] or [2024, 2025]
    total = 0
    for y in years:
        print(f"Year {y}:")
        for m in SEASON_MONTHS:
            total += pull_month(y, m)
    print(f"DONE. Total pitches cached: {total}")


if __name__ == "__main__":
    main()
