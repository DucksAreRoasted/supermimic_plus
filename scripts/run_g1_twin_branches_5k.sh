#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the completed Twin-mean checkpoint}"
MOTION_FILE="${MOTION_FILE:?Set MOTION_FILE to the comma-separated 40-motion list}"
SEED="${SEED:-42}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-g1-twin-branches-from-mean-5k-seed${SEED}}"
RUN_ROOT="${RUN_ROOT:-logs/${EXPERIMENT_LABEL}}"
mkdir -p "$RUN_ROOT"

common=(SuperMimic-GenCoord-G1 --motion-file "$MOTION_FILE" --num-envs 4096
  --device cuda:0 --env.seed "$SEED" --runner.max-iterations 5000
  --runner.log-interval 10 --runner.save-interval 1000 --runner.replay-ratio 16
  --runner.wm.num-batches 10 --runner.wm.batch-size 2048 --runner.wm.horizon 8
  --runner.policy.num-batches 1 --runner.policy.batch-size 4096 --runner.policy.horizon 32
  --runner.init-model-dir "$MODEL_DIR")

run_case() {
  local aggregation="$1"
  local kappa="$2"
  local run_name="${EXPERIMENT_LABEL}-${aggregation}"
  uv run src/supermimic/scripts/train.py "${common[@]}" --runner.critic-variant twin \
    --runner.twin-actor-aggregation "$aggregation" --runner.twin-aggregation-kappa "$kappa" \
    --runner.run-name "$run_name" 2>&1 | tee "$RUN_ROOT/${run_name}.log"
}

run_case min 1.0
run_case kappa 0.5
