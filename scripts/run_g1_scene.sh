#!/usr/bin/env bash
# Reproduce one of the pinned G1 scene evaluations; trailing options override the defaults.
set -euo pipefail
scene=${1:?expected flat, terrain, perceptive, push, uphill, or ledges}
shift
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SUPERMIMIC_DISABLE_COMPILE=1 TORCH_COMPILE_DISABLE=1
python_bin=${PYTHON_BIN:-.venv/bin/python}
source_run=logs/SuperMimic-GenCoord-G1/g1-shared-focus04-nograph-reset-safe-simweight0-warm24k-plus16k-seed42/2026-11-20_12-29-41
base=${G1_BASELINE_CHECKPOINT:-$source_run}
if [[ ! -d "$base" && -z "${G1_BASELINE_CHECKPOINT:-}" ]]; then base=../supermimic_kl_reset_safe/$source_run; fi
cross=${G1_TERRAIN_CHECKPOINT:-logs/SuperMimic-GenCoord-G1/terrain-5cm-actor-world-cross0.025-adapted-384it-seed42/2026-11-20_16-41-43}
out=${G1_SCENE_OUTPUT:-outputs/scene-demo/$scene}
case "$scene" in
  flat)
    exec "$python_bin" scripts/eval_scene_flat_regression.py "$base" \
      logs/analysis/reset_safe_nov20/protocol.json logs/analysis/reset_safe_nov20/zero_warm24k_plus16k/paired/starts.npz "$out/eval.json" "$@" ;;
  terrain)
    exec "$python_bin" scripts/eval_walk_terrain.py "$cross" outputs/cross_slope/walk2_subject4.npz "$out/eval.json" \
      --height .075 --lateral-height .03 --phase 67 --steps 200 --num-envs 48 --seed 147 --solver-iterations 60 "$@" ;;
  perceptive)
    mode=${G1_PERCEPTION:-depth-contact-stride}
    motion=${G1_WALK_MOTION:-data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk2_subject4.npz}
    exec "$python_bin" scripts/eval_walk_terrain.py "$cross" "$motion" "$out/$mode.json" \
      --perception "$mode" --height .075 --lateral-height .03 --phase 67 --steps 200 --num-envs 48 \
      --seed 147 --solver-iterations 60 --window-search-frames 3600 --starts-file "$out/starts.npz" "$@" ;;
  push|uphill|ledges)
    geometry=(); lane=flat
    if [[ $scene = uphill ]]; then geometry=(--grade .04); lane=grade; fi
    if [[ $scene = ledges ]]; then geometry=(--terrain-height .04); lane=ledges; fi
    control=()
    case ${G1_PUSH_CONTROLLER:-adaptive} in
      tracking) ;;
      paced) control=(--stop-at-goal) ;;
      adaptive)
        gains=$("$python_bin" -c 'import json; print(*json.load(open("outputs/push/gains-v2.json"))["gains"])')
        read -ra gains_array <<<"$gains"
        control=(--stop-at-goal --force-adaptive --gains "${gains_array[@]}" --policy outputs/push/policy-v2/actor_residual_100.pt) ;;
      *) echo "Unknown G1_PUSH_CONTROLLER" >&2; exit 2 ;;
    esac
    exec "$python_bin" scripts/run_push_task.py "$base" "outputs/push_release/$lane" "$out" \
      --center-x .96 --distance .68 --gap .12 --offset .22 --steps 165 --num-envs 32 --seed 1202 \
      --mass 3.6 --friction .34 "${control[@]}" "${geometry[@]}" "$@" ;;
  *) echo "Unknown G1 scene: $scene" >&2; exit 2 ;;
esac
