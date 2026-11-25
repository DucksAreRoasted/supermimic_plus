#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${1:?supply a new output directory}")
reference=$(realpath -e "${2:?supply the original 20K run manifest}")
mkdir "$out"
trap 'echo "Failed at line $LINENO" >> "$out/status"' ERR
git -C "$repo" rev-parse HEAD > "$out/commit.txt"
cp "$reference" "$out/training-reference.json"
export PYTHONPATH="$repo/src:$repo/vendor/mjlab/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl WANDB_MODE=disabled
unset SUPERMIMIC_DISABLE_COMPILE TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE
python_bin="$repo/.venv/bin/python"
protocol="$repo/docs/experiments/lafan40_seed42.json"
arms=(nominal camera oracle)

for arm in "${arms[@]}"; do
  mkdir "$out/$arm"
  cd "$out/$arm"
  echo "Training $arm" > "$out/status"
  "$python_bin" "$repo/scripts/train_g1_dual_head_depth.py" --actor-input state1 \
    --goal-width 68 --terrain-curriculum --terrain-height-reset --cache-reference \
    --camera-profile instinct --rough-reference "$arm" \
    --official-backends --num-envs 2048 --batch 2048 --policy-batch 2048 \
    --seed 42 --iterations 5000 --lafan40-protocol "$protocol" \
    --training-reference "$out/training-reference.json" > "$out/$arm.log" 2>&1
done

cd "$repo"
for arm in "${arms[@]}"; do
  echo "Evaluating $arm" > "$out/status"
  checkpoints=("$out/$arm"/logs/*/*/*/checkpoints/iter-005000)
  [[ ${#checkpoints[@]} == 1 && -d ${checkpoints[0]} ]]
  start_args=()
  if [[ $arm == nominal ]]; then start_args+=(--save-starts); fi
  "$python_bin" scripts/eval_scene_flat_regression.py "${checkpoints[0]}" "$protocol" \
    "$out/starts.npz" "$out/$arm.json" --rough-seed 1702 --rough-scale 0.25 \
    --leg-ablation on "${start_args[@]}" > "$out/eval-$arm.log" 2>&1
done
"$python_bin" scripts/summarize_flat_depth_5k.py "$out" "${arms[@]}" > "$out/comparison.json"
echo "Completed training and paired evaluations" > "$out/status"
