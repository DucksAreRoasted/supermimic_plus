#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the completed Twin-mean checkpoint}"
MOTION_FILE="${MOTION_FILE:?Set MOTION_FILE to the comma-separated 40-motion list}"
RUN_ROOT="${RUN_ROOT:-logs/g1_twin_branches_5k}"
mkdir -p "$RUN_ROOT"

common=(SuperMimic-GenCoord-G1 --motion-file "$MOTION_FILE" --num-envs 4096
  --device cuda:0 --env.seed 42 --runner.max-iterations 5000
  --runner.log-interval 10 --runner.save-interval 1000 --runner.replay-ratio 16
  --runner.wm.num-batches 10 --runner.wm.batch-size 2048 --runner.wm.horizon 8
  --runner.policy.num-batches 1 --runner.policy.batch-size 4096 --runner.policy.horizon 32
  --runner.init-model-dir "$MODEL_DIR")

uv run src/supermimic/scripts/train.py "${common[@]}" --runner.critic-variant twin \
  --runner.twin-actor-aggregation min --runner.twin-aggregation-kappa 1.0 \
  --runner.run-name g1-twin-min-branch5k-seed42
uv run src/supermimic/scripts/train.py "${common[@]}" --runner.critic-variant twin \
  --runner.twin-actor-aggregation kappa --runner.twin-aggregation-kappa 0.5 \
  --runner.run-name g1-twin-kappa-branch5k-seed42
