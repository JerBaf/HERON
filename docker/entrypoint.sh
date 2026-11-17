#!/usr/bin/env bash
# Container entry point: a CPU smoke test, then training and test evaluation of HERON on the GPU.
#   entrypoint.sh <dataset>      e.g. network: reads the windows in /data/<dataset>, writes runs/<dataset>
# Optional environment: DATA_ROOT (window directory), OUT_DIR (output directory), EPOCHS.
set -euo pipefail

dataset=${1:?usage: entrypoint.sh <dataset>, e.g. network}
data_root=${DATA_ROOT:-/data/$dataset}
out_dir=${OUT_DIR:-runs/$dataset}
epochs=${EPOCHS:-30}

cd /app
export PYTHONPATH=/app      # the scripts import plug-ins from the repository root
echo "== [1/2] smoke test (CPU, toy data)"
python scripts/smoke_test.py

echo "== [2/2] training and test evaluation on $dataset (GPU, $epochs epochs)"
python scripts/run_test.py --config "configs/$dataset.yaml" \
    data.root="$data_root" out_dir="$out_dir" train.device=cuda train.epochs="$epochs"

metrics="$out_dir/test/metrics.json"
ade=$(jq -r '.ade' "$metrics")
min_fde=$(jq -r '.min_fde' "$metrics")
macro_f1=$(jq -r '.macro_f1' "$metrics")
echo "HERON pipeline completed: test ADE=$ade  minFDE=$min_fde  macro-F1=$macro_f1"
