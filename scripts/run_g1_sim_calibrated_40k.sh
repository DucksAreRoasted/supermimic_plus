#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to a completed checkpoint}"
MOTION_FILE="${MOTION_FILE:?Set MOTION_FILE to the comma-separated 40-motion list}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-g1-sim-calibrated-lafan40-40kit-seed${SEED}}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"
mkdir -p "$(dirname -- "$LOG_FILE")"

uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file "$MOTION_FILE" --num-envs 4096 --device cuda:0 --env.seed "$SEED" \
  --runner.max-iterations 40000 --runner.log-interval 10 --runner.save-interval 1000 \
  --runner.replay-ratio 16 --runner.wm.num-batches 10 --runner.wm.batch-size 2048 \
  --runner.wm.horizon 8 --runner.policy.num-batches 1 --runner.policy.batch-size 4096 \
  --runner.policy.horizon 32 --runner.critic-variant sim-calibrated \
  --runner.sim-calibration-weight 0.1 --runner.sim-calibration-horizon 32 \
  --runner.sim-calibration-interval 1 --runner.init-model-dir "$MODEL_DIR" \
  --runner.run-name "$RUN_NAME" 2>&1 | tee "$LOG_FILE"
