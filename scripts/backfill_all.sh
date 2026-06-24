#!/usr/bin/env bash
set -euo pipefail
echo "=== MLB K-Predictor: Full Historical Backfill ==="
echo "This will pull Statcast + MLB API data for 2022-2025."
echo "Estimated time: ~4 hours (rate-limited)."
echo ""
python -m src.data.backfill --start-year 2022 --end-year 2025
echo "=== Backfill complete ==="
