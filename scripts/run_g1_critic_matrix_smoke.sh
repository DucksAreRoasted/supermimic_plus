#!/usr/bin/env bash
set -euo pipefail

# Small, fail-fast matrix smoke test. Override SMOKE_NUM_ENVS/SMOKE_ITERS if needed.
NUM_ENVS="${SMOKE_NUM_ENVS:-8}"
MAX_ITERATIONS="${SMOKE_ITERS:-1}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-7}"
MOTION_FILE="${MOTION_FILE:-data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk1_subject1.npz}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-g1-critic-matrix-smoke-${MAX_ITERATIONS}it-seed${SEED}}"
RUN_ROOT="${RUN_ROOT:-logs/${EXPERIMENT_LABEL}}"

[[ -f "$MOTION_FILE" ]] || { echo "Missing motion file: $MOTION_FILE" >&2; exit 1; }
mkdir -p "$RUN_ROOT"

common=(
  SuperMimic-GenCoord-G1 --motion-file "$MOTION_FILE"
  --num-envs "$NUM_ENVS" --device "$DEVICE" --env.seed "$SEED"
  --runner.max-iterations "$MAX_ITERATIONS" --runner.log-interval 1
  --runner.save-interval 100 --runner.wm.num-batches 1 --runner.wm.batch-size "$NUM_ENVS"
  --runner.wm.horizon 2 --runner.policy.num-batches 1 --runner.policy.batch-size "$NUM_ENVS"
  --runner.policy.horizon 2
)

run_case() {
  local name="$1" aggregation="$2" kappa="$3"
  echo "=== $name ==="
  uv run src/supermimic/scripts/train.py "${common[@]}" \
    --runner.critic-variant twin \
    --runner.twin-actor-aggregation "$aggregation" \
    --runner.twin-aggregation-kappa "$kappa" \
    --runner.run-name "${EXPERIMENT_LABEL}-${name}" 2>&1 | tee "$RUN_ROOT/${EXPERIMENT_LABEL}-${name}.log"
  grep -q "Learning iteration" "$RUN_ROOT/${EXPERIMENT_LABEL}-${name}.log"
  grep -q "Saved checkpoint" "$RUN_ROOT/${EXPERIMENT_LABEL}-${name}.log"
}

run_case twin-mean mean 0.0
run_case twin-min min 1.0
run_case twin-kappa kappa 0.5

echo "=== sim-calibrated ==="
uv run src/supermimic/scripts/train.py "${common[@]}" \
  --runner.critic-variant sim-calibrated \
  --runner.sim-calibration-weight 0.1 \
  --runner.sim-calibration-horizon 2 \
  --runner.run-name "${EXPERIMENT_LABEL}-sim-calibrated" 2>&1 | tee "$RUN_ROOT/${EXPERIMENT_LABEL}-sim-calibrated.log"
grep -q "Learning iteration" "$RUN_ROOT/${EXPERIMENT_LABEL}-sim-calibrated.log"
grep -q "Saved checkpoint" "$RUN_ROOT/${EXPERIMENT_LABEL}-sim-calibrated.log"
