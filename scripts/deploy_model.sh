#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_PATH=${1:?Usage: deploy_model.sh <path_to_experiment_model>}
echo "Promoting $MODEL_PATH to production..."
cp "$MODEL_PATH" data/models/production/
echo "Model deployed to data/models/production/"
