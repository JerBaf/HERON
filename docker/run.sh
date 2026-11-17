#!/usr/bin/env bash
# Train and evaluate HERON on the GPU, with the downloaded windows mounted read-only.
set -euo pipefail
cd "$(dirname "$0")/../data"
docker run --rm --gpus all \
    -v network:/data/network:ro \
    heron "$@"
