#!/usr/bin/env bash
# Run once in a fresh output directory on a single GPU.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${1:?supply a new output directory}")
mkdir "$out"
cd "$out"
trap 'echo "Failed at line $LINENO; see per-arm logs" >> "$out/status"' ERR
export PYTHONPATH="$repo/src:$repo/vendor/mjlab/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl WANDB_MODE=disabled
export SUPERMIMIC_DISABLE_COMPILE=1 TORCH_COMPILE_DISABLE=1
python_bin="$repo/.venv/bin/python"
protocol="$repo/docs/experiments/lafan40_seed42.json"
git -C "$repo" rev-parse HEAD > commit.txt
for arm in base aux full; do
  echo "Training $arm" > status
  "$python_bin" "$repo/scripts/train_g1_dual_head_depth.py" --flat-depth-arm "$arm" \
    --lafan40-protocol "$protocol" --cache-reference --official-backends --goal-width 68 \
    --actor-input proprio1 --seed 42 --iterations 5000 --num-envs 4096 \
    --batch 2048 --policy-batch 4096 --actor-microbatch 2048 > "$arm.log" 2>&1
done
cd "$repo"
for arm in base aux full; do
  echo "Evaluating $arm" > "$out/status"
  checkpoints=("$out"/logs/*/g1-flat-"$arm"-*/*/checkpoints/iter-005000)
  args=()
  if [[ $arm == base ]]; then args+=(--save-starts); fi
  "$python_bin" scripts/eval_scene_flat_regression.py "${checkpoints[0]}" "$protocol" \
    "$out/starts.npz" "$out/$arm.json" "${args[@]}" > "$out/eval-$arm.log" 2>&1
done
"$python_bin" scripts/summarize_flat_depth_5k.py "$out" > "$out/comparison.json"
echo 'Completed training and paired evaluations' > "$out/status"
