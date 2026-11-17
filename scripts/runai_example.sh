#!/bin/bash
# Cross-validation and final test of HERON on one paper dataset, inside a runAI job (1 GPU).
#   bash scripts/runai_example.sh network [section.key=value ...]     # or linear / gompertz
set -euo pipefail
REPO=/scratch/data/HERON
cd "$REPO/official"
export PYTHONPATH="$PWD"
PY="$REPO/reproducibility/env/heron/bin/python"
DATASET=${1:-network}                          # linear / gompertz / network: data root comes from the config
shift || true                                  # remaining arguments: section.key=value overrides
OUT="runs/$DATASET"
mkdir -p "$OUT"
"$PY" scripts/run_cv.py   --config "configs/$DATASET.yaml" out_dir="$OUT" "$@" 2>&1 | tee "$OUT/cv.log"
"$PY" scripts/run_test.py --config "configs/$DATASET.yaml" out_dir="$OUT" "$@" 2>&1 | tee "$OUT/test.log"
