#!/usr/bin/env bash
# Run once in a fresh output directory on a single GPU.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${1:?supply a new output directory}")
# Passing an archived training manifest selects the three base-Actor input arms.
arms=(base aux full)
reference_args=()
if [[ $# == 2 ]]; then
  reference_args=(--training-reference "$(realpath -e "$2")")
  arms=(state1 proprio10 proprio97x10)
fi
mkdir "$out"
cd "$out"
trap 'echo "Failed at line $LINENO; see per-arm logs" >> "$out/status"' ERR
export PYTHONPATH="$repo/src:$repo/vendor/mjlab/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl WANDB_MODE=disabled
export SUPERMIMIC_DISABLE_COMPILE=1 TORCH_COMPILE_DISABLE=1
python_bin="$repo/.venv/bin/python"
protocol="$repo/docs/experiments/lafan40_seed42.json"
git -C "$repo" rev-parse HEAD > commit.txt
for arm in "${arms[@]}"; do
  fusion=$arm
  layout=proprio1
  if (( ${#reference_args[@]} )); then
    fusion=base
    layout=$arm
    mkdir "$out/$arm"
    cd "$out/$arm"
  fi
  echo "Training $arm" > "$out/status"
  "$python_bin" "$repo/scripts/train_g1_dual_head_depth.py" --flat-depth-arm "$fusion" "${reference_args[@]}" \
    --lafan40-protocol "$protocol" --cache-reference --official-backends --goal-width 68 \
    --actor-input "$layout" --seed 42 --iterations 5000 --num-envs 4096 \
    --batch 2048 --policy-batch 4096 --actor-microbatch 2048 > "$out/$arm.log" 2>&1
done
cd "$repo"
for arm in "${arms[@]}"; do
  echo "Evaluating $arm" > "$out/status"
  run_dir="$out"
  if (( ${#reference_args[@]} )); then run_dir="$out/$arm"; fi
  checkpoints=("$run_dir"/logs/*/g1-flat-*/*/checkpoints/iter-005000)
  if [[ $run_dir == "$out" ]]; then
    checkpoints=("$out"/logs/*/g1-flat-"$arm"-*/*/checkpoints/iter-005000)
  fi
  [[ ${#checkpoints[@]} == 1 && -d ${checkpoints[0]} ]]
  args=()
  if [[ $arm == "${arms[0]}" ]]; then args+=(--save-starts); fi
  "$python_bin" scripts/eval_scene_flat_regression.py "${checkpoints[0]}" "$protocol" \
    "$out/starts.npz" "$out/$arm.json" "${args[@]}" > "$out/eval-$arm.log" 2>&1
done
"$python_bin" scripts/summarize_flat_depth_5k.py "$out" "${arms[@]}" > "$out/comparison.json"
echo 'Completed training and paired evaluations' > "$out/status"
