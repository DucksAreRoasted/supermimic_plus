#!/usr/bin/env bash
# Run one server experiment through its physical evaluation.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
dest=outputs/dual-head-depth/scratch-env4096
iterations=40000
profile=()
wm_lr=2e-5
control_dir=
smoke_init=()
waypoint_args=()
if [[ "${1:-}" == "--fast-20k" ]]; then
  shift
  dest=outputs/dual-head-depth/scratch-env4096-b1024-refcache-20k
  iterations=20000
  profile=(--batch 1024 --official-backends --cache-reference)
  if [[ "${1:-}" == "--wm-lr" && $# == 2 ]]; then
    wm_lr=$(python3 - "$2" <<'PY'
import math, sys
rate = float(sys.argv[1])
if not math.isfinite(rate) or rate <= 0 or rate == 2e-5:
    sys.exit("Choose a distinct, positive, finite world-model learning rate")
print(repr(rate))
PY
    )
    control_dir="$dest"
    dest="$dest-wmlr$wm_lr"
    profile+=(--wm-lr "$wm_lr")
    smoke_init=(--expect-initial-hashes "$control_dir/initial-parameters.json")
    shift 2
  elif [[ "${1:-}" == "--root-waypoints" && $# == 1 ]]; then
    control_dir="$dest"
    dest="$dest-waypoints"
    waypoint_args=(--root-waypoints)
    profile+=(--root-waypoints --expect-control-hashes "$control_dir/initial-parameters.json")
    shift
  fi
fi
if (( $# != 0 )); then
  echo "Usage: $0 [--fast-20k [--wm-lr RATE | --root-waypoints]]" >&2
  exit 2
fi
formal_log="$dest/formal-$((iterations / 1000))k.log"
mkdir -p "$dest"
exec 9>"$dest/launch.lock"
flock -n 9
test ! -e "$formal_log"
export PYTHONPATH="$PWD/src:$PWD/vendor/mjlab/src" MUJOCO_GL=egl OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled SUPERMIMIC_DISABLE_COMPILE=1 TORCH_COMPILE_DISABLE=1
py=.venv/bin/python
checkpoint() {
  local manifest
  manifest=$(sed -n 's/^\[INFO\] Run manifest: //p' "$1")
  test -f "$manifest"
  dirname "$manifest"
}

if [[ -n "$control_dir" ]]; then
  for file in "$control_dir/initial-parameters.json" "$control_dir/holdout-20k/summary.json" \
      "$control_dir/followup/repeat-controls/primary/summary.json" \
      outputs/foot-depth-pilot/holdout/summary.json outputs/dual-head-depth/independent-reference/summary.json; do
    test -s "$file" || { echo "Missing 20K control: $file" >&2; exit 1; }
  done
fi
if [[ ! -f "$dest/smoke.log" ]]; then
  "$py" scripts/train_g1_dual_head_depth.py --num-envs 4096 --iterations 40 "${profile[@]}" "${smoke_init[@]}" >"$dest/smoke.log" 2>&1
fi
smoke="$(checkpoint "$dest/smoke.log")/checkpoints/iter-000040"
if [[ ! -f "$dest/smoke-eval.json" ]]; then
  "$py" scripts/eval_walk_terrain.py "$smoke" data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk1_subject5.npz "$dest/smoke-eval.json" \
    --perception depth --num-envs 8 --steps 60 --seed 271 --landscape-seed 701 --height .09 --lateral-height .04 \
    --solver-iterations 60 --window-search-frames 3600 --depth-noise-m .005 --depth-dropout .15 >"$dest/smoke-eval.log" 2>&1
fi
control_smoke=
if [[ -n "$control_dir" ]]; then
  control_smoke="$(checkpoint "$control_dir/smoke.log")/checkpoints/iter-000040"
fi
if (( ${#waypoint_args[@]} )); then
  "$py" scripts/probe_depth_path.py "$smoke" outputs/foot-depth-pilot/banks/walk1_subject5/bank.pt "$dest/waypoint-smoke.json" >"$dest/waypoint-smoke.log" 2>&1
fi
"$py" - "$smoke" "$dest" "$iterations" "$wm_lr" "$control_smoke" "$control_dir" "${#waypoint_args[@]}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, "scripts")
from follow_depth_result import check_control_config, check_fast_profile, check_scratch

checkpoint, dest = map(Path, sys.argv[1:3])
waypoints = bool(int(sys.argv[7]))
check_scratch(checkpoint, 40, waypoints)
if int(sys.argv[3]) == 20000:
  check_fast_profile(checkpoint, float(sys.argv[4]))
if sys.argv[5]:
  check_control_config(checkpoint, Path(sys.argv[5]), root_waypoints=waypoints)
hashes, = [json.loads(line)["initial_parameter_sha256"]
           for line in (dest / "smoke.log").read_text().splitlines()
           if line.startswith('{"initial_parameter_sha256"')]
assert set(hashes) == {"actor", "depth", "dynamics", "critic"}
if sys.argv[6]:
  common = hashes
  if waypoints:
    common, = [json.loads(line)["control_parameter_sha256"]
               for line in (dest / "smoke.log").read_text().splitlines()
               if line.startswith('{"control_parameter_sha256"')]
    probe = json.loads((dest / "waypoint-smoke.json").read_text())
    assert probe["use_root_waypoints"] and probe["actor_sha256"] == hashlib.sha256((checkpoint/"actor.pt").read_bytes()).hexdigest()
    assert len(probe["offsets"]) == 4 and all(r["action_mean_abs"] > 0 for r in probe["offsets"])
  assert common == json.loads((Path(sys.argv[6]) / "initial-parameters.json").read_text())
for filename in ("actor.pt", "world_model.pt", "critic.pt"):
  state = torch.load(checkpoint / filename, weights_only=True, map_location="cpu")
  if filename == "world_model.pt":
    state = state["dynamics"]
    assert all(state[f"head_mlps.{i}.2.weight"].norm() > 0 for i in (0, 1))
  if filename == "actor.pt":
    assert state["foot_depth_fusion.modulator.2.weight"].norm() > 0
  assert all(t.isfinite().all() for t in state.values())
evaluation = json.loads((dest / "smoke-eval.json").read_text())
assert evaluation["fusion_sha256"] == hashlib.sha256((checkpoint / "actor.pt").read_bytes()).hexdigest()
assert evaluation["windows"] == len(evaluation["windows_detail"]) == 8
assert evaluation["perception"] == "depth" and not evaluation["external_joint_compensation"]
assert all(r["steps"] > 0 and math.isfinite(r["mean_body_error_cm"]) and
           math.isfinite(r["reward_per_planned_step"]) for r in evaluation["windows_detail"])
(dest / "initial-parameters.json").write_text(json.dumps(hashes, indent=2) + "\n")
PY
echo "$(date -Is) scratch smoke and physical reload passed"
"$py" scripts/train_g1_dual_head_depth.py --num-envs 4096 --iterations "$iterations" --seed 42 "${profile[@]}" --expect-initial-hashes "$dest/initial-parameters.json" >"$formal_log" 2>&1
printf -v final_step "%06d" "$iterations"
trained="$(checkpoint "$formal_log")/checkpoints/iter-$final_step"
echo "$(date -Is) scratch training finished; evaluating"
historical=logs/SuperMimic-GenCoord-G1/terrain-5cm-actor-world-cross0.025-adapted-384it-seed42/2026-11-20_16-41-43
adapter=outputs/foot-depth-pilot/pilot/concat.pt
if [[ -z "$control_dir" ]]; then
  "$py" scripts/eval_g1_dual_head_depth.py "$historical" outputs/foot-depth-pilot/holdout --panel primary --baseline-adapter "$adapter" >"$dest/primary-reference.log" 2>&1
  "$py" scripts/eval_g1_dual_head_depth.py "$historical" outputs/dual-head-depth/independent-reference --panel independent --baseline-adapter "$adapter" >"$dest/independent-reference.log" 2>&1
else
  control="$(checkpoint "$control_dir/formal-20k.log")/checkpoints/iter-$final_step"
  "$py" - "$trained" "$control" "${#waypoint_args[@]}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from follow_depth_result import check_control_config
check_control_config(*map(Path, sys.argv[1:3]), root_waypoints=bool(int(sys.argv[3])))
PY
  mkdir -p "$dest/followup"
  cp -a "$control_dir/followup/repeat-controls" "$dest/followup/"
fi
"$py" scripts/eval_g1_dual_head_depth.py "$trained" "$dest/holdout-$((iterations / 1000))k" >"$dest/final-eval.log" 2>&1
"$py" scripts/follow_depth_result.py "$trained" "$dest/holdout-$((iterations / 1000))k" "$dest/followup" --iterations "$iterations" --wm-lr "$wm_lr" "${waypoint_args[@]}" >"$dest/followup.log" 2>&1
if [[ -n "$control_dir" ]]; then
  "$py" - "$control_dir" "$dest" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from follow_depth_result import HELD_OUT, compare, record
control, candidate = map(Path, sys.argv[1:])
reports = {}
for panel in ("holdout-20k", "holdout-20k-repeat",
              "followup/original-independent", "followup/original-independent-repeat"):
  if all((d/panel/"summary.json").is_file() for d in (control, candidate)):
    reports[panel] = compare(control/panel, candidate/panel, HELD_OUT, "depth.json")
record(candidate/"followup/versus-original-20k.json", reports)
PY
fi
echo "$(date -Is) evaluation and follow-up complete"
