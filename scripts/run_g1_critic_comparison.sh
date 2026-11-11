#!/usr/bin/env bash
set -euo pipefail

# Override these for the target server.
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-data/modelscope/LAFAN1_Retargeting_Dataset}"
CLIP="${CLIP:-walk1_subject1}"
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

common_args=(
  SuperMimic-GenCoord-G1
  --motion-file "$MOTION_FILE"
  --num-envs "$NUM_ENVS"
  --device "$DEVICE"
  --runner.max-iterations "$MAX_ITERATIONS"
  --runner.log-interval 100
  --runner.save-interval "$MAX_ITERATIONS"
  --runner.replay-ratio 16
  --runner.wm.num-batches 1
  --runner.wm.batch-size 4096
  --runner.wm.horizon 32
  --runner.policy.num-batches 1
  --runner.policy.batch-size 4096
  --runner.policy.horizon 32
)

export SUPERMIMIC_DISABLE_COMPILE="${SUPERMIMIC_DISABLE_COMPILE:-0}"
mkdir -p "$RUN_ROOT"

echo "=== Original single critic ==="
uv run src/supermimic/scripts/train.py "${common_args[@]}" \
  --runner.policy.twin-critic False \
  --runner.run-name g1-single-critic

echo "=== Stage 1 twin critic ==="
uv run src/supermimic/scripts/train.py "${common_args[@]}" \
  --runner.policy.twin-critic True \
  --runner.run-name g1-twin-critic

echo "TensorBoard: tensorboard --logdir logs"
