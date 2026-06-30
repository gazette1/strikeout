"""
snapshot_state.py — Precompute compact pitcher/team state from the full Statcast
cache and write it to data/state/ (small, committed to git).

This lets the cloud run (daily_picks.py) load current pitcher/team state instantly
without backfilling ~290MB of pitch data at runtime. Regenerate locally whenever you
refresh the cache (e.g. weekly), then commit data/state/.

Run: python tools/snapshot_state.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from tools.build_dataset import load_pitches
from tools.daily_picks import pitcher_state, team_state, norm

STATE = ROOT / "data" / "state"
STATE.mkdir(parents=True, exist_ok=True)

KEEP = ["pitcher", "player_name", "name_key", "p_throws", "k_pct_career", "swstr_pct",
        "csw_pct", "putaway_rate", "bf_per_start", "k_last3", "fb_velo_asof",
        "n_prior_starts", "last_date"]


def main():
    pitches = load_pitches()
    ps = pitcher_state(pitches).reset_index()
    ps["name_key"] = ps["player_name"].map(norm)
    ps = ps[KEEP]
    ps.to_parquet(STATE / "pitcher_state.parquet", index=False)

    opp_k, opp_k_hand = team_state(pitches)
    opp_k.reset_index().to_parquet(STATE / "team_overall.parquet", index=False)
    opp_k_hand.reset_index().to_parquet(STATE / "team_vs_hand.parquet", index=False)

    meta = {"snapshot_date": str(ps["last_date"].max().date()),
            "n_pitchers": int(len(ps)), "n_teams": int(len(opp_k))}
    (STATE / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Snapshot written -> {STATE}")
    print(f"  {meta['n_pitchers']} pitchers, {meta['n_teams']} teams, "
          f"as of {meta['snapshot_date']}")
    total = sum(f.stat().st_size for f in STATE.glob('*'))
    print(f"  total size: {total/1024:.0f} KB")


if __name__ == "__main__":
    main()
