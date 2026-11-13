#!/usr/bin/env bash
set -euo pipefail

# Override these for the target server.
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-data/modelscope/LAFAN1_Retargeting_Dataset}"
CLIP="${CLIP:-*}"
RUN_ROOT="${RUN_ROOT:-logs/g1_critic_comparison}"

MOTION_FILE="$DATA_ROOT/npz/$CLIP.npz"
if [[ "$MOTION_FILE" == *'*'* || "$MOTION_FILE" == *'?'* ]]; then
  compgen -G "$MOTION_FILE" >/dev/null || {
    echo "No motion files matched $MOTION_FILE; run scripts/prepare_lafan_g1.py first." >&2
    exit 1
  }
elif [[ ! -f "$MOTION_FILE" ]]; then
  echo "Missing $MOTION_FILE; run scripts/prepare_lafan_g1.py first." >&2
  exit 1
fi

# Pass one parser argument regardless of how many files the glob matches.
# tyro accepts a comma-separated motion_file and train.py expands each entry.
mapfile -t MATCHED_MOTIONS < <(compgen -G "$MOTION_FILE" | sort)
if (( ${#MATCHED_MOTIONS[@]} == 0 )); then
  echo "No motion files matched $MOTION_FILE" >&2
  exit 1
fi
MOTION_FILE_CSV=$(IFS=,; echo "${MATCHED_MOTIONS[*]}")

common_args=(
  SuperMimic-GenCoord-G1
  --motion-file "$MOTION_FILE_CSV"
  --num-envs "$NUM_ENVS"
  --device "$DEVICE"
  --env.seed "$SEED"
  --runner.max-iterations "$MAX_ITERATIONS"
  --runner.log-interval 10
  --runner.save-interval 1000
  --runner.replay-ratio 16
  --runner.wm.num-batches 10
  --runner.wm.batch-size 2048
  --runner.wm.horizon 8
  --runner.policy.num-batches 1
  --runner.policy.batch-size 4096
  --runner.policy.horizon 32
)

export SUPERMIMIC_DISABLE_COMPILE="${SUPERMIMIC_DISABLE_COMPILE:-0}"
mkdir -p "$RUN_ROOT"

echo "Motion files: $MOTION_FILE (one shared policy across all matched clips)"
echo "Matched training budget: $MAX_ITERATIONS iterations per variant; seed=$SEED"
echo "=== Official single critic: unmodified MSE + value/return clamp ==="
uv run src/supermimic/scripts/train.py "${common_args[@]}" \
  --runner.critic-variant official \
  --runner.run-name g1-official-single-critic-lafan

echo "=== Stage 1 twin critic ==="
uv run src/supermimic/scripts/train.py "${common_args[@]}" \
  --runner.critic-variant twin \
  --runner.run-name g1-twin-critic-lafan

echo "TensorBoard: tensorboard --logdir logs"
