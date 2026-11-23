#!/usr/bin/env bash
# Run the matched flat-ground pair on one GPU and replay both final policies.
set -euo pipefail
cd "$(dirname "$0")/.."
protocol=${1:?Supply the LAFAN40 protocol JSON}
starts=${2:?Supply its paired flat-ground starts.npz}
out=${3:?Supply a new output directory}
test -s "$protocol" && test -s "$starts"
mkdir "$out"
exec > "$out/queue.log" 2>&1
export PYTHONPATH="$PWD/src:$PWD/vendor/mjlab/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_MODE=disabled
python="$PWD/.venv/bin/python"
stage=starting
trap 'echo "$(date -Is) failed at $stage" > "$out/status"' ERR
sha256sum "$protocol" "$starts" > "$out/evaluation-inputs.sha256"
for head in nonlinear linear; do
  stage="train-$head"
  echo "$(date -Is) $stage" | tee "$out/status"
  log="$out/$head-train.log"
  "$python" scripts/train_g1_flat_head_ablation.py "$head" --protocol "$protocol" \
    --iterations 20000 --initial-parameters "$out/initial-parameters.json" > "$log" 2>&1
  manifest=$(sed -n 's/^\[INFO\] Run manifest: //p' "$log" | tail -1)
  checkpoint="$(dirname "$manifest")/checkpoints/iter-020000"
  for file in run_manifest.json actor.pt world_model.pt critic.pt; do test -s "$checkpoint/$file"; done
  printf '%s\n' "$checkpoint" > "$out/$head-checkpoint.txt"
done
for head in nonlinear linear; do
  stage="eval-$head"
  echo "$(date -Is) $stage" | tee "$out/status"
  "$python" scripts/eval_scene_flat_regression.py "$(cat "$out/$head-checkpoint.txt")" \
    "$protocol" "$starts" "$out/$head-eval.json" > "$out/$head-eval.log" 2>&1
done
echo "$(date -Is) completed" | tee "$out/status"
