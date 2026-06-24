#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATE=${1:-$(date +%Y-%m-%d)}
echo "=== Daily K Predictions for $DATE ==="
python -m src.pipeline.daily_runner --date "$DATE"
echo "=== Done ==="
