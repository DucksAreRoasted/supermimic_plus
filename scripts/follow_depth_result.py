"""Confirm a completed depth run and, on measured path drift, retrain with observable waypoints."""

import argparse
import fcntl
import hashlib
import json
import re
import shlex
from pathlib import Path

from compare_depth_runs import compare
from run_foot_depth_pilot import BASE, ROOT, run
from validate_depth_mimic import CASES

PRIMARY = ROOT / "outputs/foot-depth-pilot/holdout"
INDEPENDENT = ROOT / "outputs/dual-head-depth/independent-reference"
BANK = ROOT / "outputs/foot-depth-pilot/banks/walk1_subject5/bank.pt"
HELD_OUT = [
  case[0]
  for suite in ("confirmation", "prospective")
  for case in CASES[suite]
  if case[0] != "walk1_subject1"
]


def read(path):
  return json.loads(path.read_text())


def record(path, value):
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  tmp.replace(path)


def validate_panel(checkpoint, directory, reference):
  if (
    Path(read(directory / "summary.json")["checkpoint"]).resolve()
    != checkpoint.resolve()
  ):
    raise ValueError("Result belongs to a different checkpoint")
  actor_sha = hashlib.sha256((checkpoint / "actor.pt").read_bytes()).hexdigest()
  for name in HELD_OUT:
    if read(directory / name / "depth.json")["fusion_sha256"] != actor_sha:
      raise ValueError("Actor identity changed during evaluation")
  report = compare(reference, directory, HELD_OUT)
  return {k: v for k, v in report.items() if k != "transitions"}


def evaluate(checkpoint, directory, panel, reference=None):
  if reference is None:
    reference = PRIMARY if panel == "primary" else INDEPENDENT
  if not (directory / "summary.json").is_file():
    run(
      "eval_g1_dual_head_depth.py",
      [checkpoint, directory, "--panel", panel, "--paired-starts", reference],
      directory / "eval.log",
    )
  return validate_panel(checkpoint, directory, reference)


def repeat_panel(checkpoint, directory, panel, reference, controls):
  control = controls / panel
  adapter = ROOT / "outputs/foot-depth-pilot/pilot/concat.pt"
  if not (control / "summary.json").is_file():
    run("eval_g1_dual_head_depth.py",
        [BASE, control, "--panel", panel, "--paired-starts", reference, "--baseline-adapter", adapter],
        control / "eval.log")
  adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
  for ref in (reference, control):
    if Path(read(ref / "summary.json")["checkpoint"]).resolve() != BASE.resolve():
      raise ValueError(f"Historical control checkpoint differs in {ref}")
    for clip in HELD_OUT:
      if read(ref / clip / "concat.json")["fusion_sha256"] != adapter_sha:
        raise ValueError(f"Historical adapter differs in {ref / clip}")
  for clip in HELD_OUT:
    if (reference / clip / "starts.npz").read_bytes() != (control / clip / "starts.npz").read_bytes():
      raise ValueError(f"Repeated physical starts differ for {clip}")
  return evaluate(checkpoint, directory.with_name(directory.name + "-repeat"), panel, control)


def verify(checkpoint, primary, independent):
  result = dict(checkpoint=str(checkpoint), confirmed=False, repeat_inconclusive=False)
  controls = independent.parent / "repeat-controls"
  for panel, directory, reference in (("primary", primary, PRIMARY),
                                      ("independent", independent, INDEPENDENT)):
    first = (validate_panel(checkpoint, directory, reference) if panel == "primary"
             else evaluate(checkpoint, directory, panel))
    second = repeat_panel(checkpoint, directory, panel, reference, controls)
    result[panel] = {**first, "repeat": second}
    if any(first[k] != second[k] for k in ("pass_rate_met", "tracking_met")):
      result["repeat_inconclusive"] = True
      return result
    if not first["meets_criteria"]:
      return result
  result["confirmed"] = True
  return result


def check_scratch(checkpoint, iterations, root_waypoints, *, terrain_height_reset=False):
  manifest = read(checkpoint / "run_manifest.json")
  cfg = manifest["runner_cfg"]
  if (
    manifest["iteration"] != iterations or cfg["max_iterations"] != iterations
    or manifest["seed"] != 42 or manifest["num_envs"] != 4096
    or cfg["use_root_waypoints"] != root_waypoints
    or cfg.get("terrain_height_reset", False) != terrain_height_reset
    or cfg["depth_fusion"] != "concat" or cfg["shared_head_width"] != 128
    or any(cfg[k] is not None for k in ("init_model_dir", "init_depth_adapter", "wandb_run_path"))
    or any(manifest[k] is not None for k in ("base_sha256", "adapter_sha256"))
  ):
    raise RuntimeError(f"Unexpected initialization or experiment configuration in {checkpoint}")


def train_waypoints(directory, iterations):
  directory.mkdir(parents=True, exist_ok=True)
  pointer = directory / "checkpoint.json"
  if not pointer.exists():
    log = directory / "train.log"
    run(
      "train_g1_dual_head_depth.py",
      ["--iterations", iterations, "--num-envs", 4096, "--seed", 42, "--root-waypoints"],
      log,
    )
    paths = re.findall(r"^\[INFO\] Run manifest: (.+)$", log.read_text(), re.MULTILINE)
    if len(paths) != 1:
      raise RuntimeError("Training returned without exactly one run manifest")
    checkpoint = (ROOT / paths[0]).parent / "checkpoints" / f"iter-{iterations:06d}"
  else:
    checkpoint = Path(read(pointer)["path"])
  check_scratch(checkpoint, iterations, True)
  record(pointer, dict(path=str(checkpoint.resolve())))
  return checkpoint.resolve()


def check_fast_profile(checkpoint, wm_lr=2e-5):
  manifest = read(checkpoint / "run_manifest.json")
  cfg = manifest["runner_cfg"]
  if not (
    cfg["wm"]["batch_size"] == cfg["policy"]["batch_size"] == 1024
    and cfg["buffer_size"] == 512 and cfg["cache_motion_reference"]
    and cfg["wm"]["lr"] == wm_lr and cfg["policy"]["lr"] == 2e-5
    and "--official-backends" in shlex.split(manifest["command"])
  ):
    raise ValueError(f"Unexpected 20K speed configuration in {checkpoint}")


def check_control_config(checkpoint, control, *, root_waypoints=False):
  candidate = read(checkpoint / "run_manifest.json")
  baseline = read(control / "run_manifest.json")
  for field in ("motion_files", "seed", "num_envs", "iteration", "task_id",
                "base_sha256", "adapter_sha256", "representation"):
    if candidate[field] != baseline[field]:
      raise ValueError(f"Comparison with the 20K control differs in {field}")
  cfg, reference_cfg = candidate["runner_cfg"], baseline["runner_cfg"]
  if root_waypoints:
    if not cfg.pop("use_root_waypoints") or reference_cfg.pop("use_root_waypoints"):
      raise ValueError("Expected only the candidate to receive root waypoints")
  elif cfg["wm"].pop("lr") == reference_cfg["wm"].pop("lr"):
    raise ValueError("Expected different world-model learning rates")
  cfg.pop("run_name")
  reference_cfg.pop("run_name")
  if cfg != reference_cfg:
    raise ValueError("Another runner configuration changed in the 20K comparison")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("primary", type=Path)
  parser.add_argument("out", type=Path)
  parser.add_argument("--iterations", type=int, choices=(20000, 40000), default=40000)
  parser.add_argument("--wm-lr", type=float, default=2e-5)
  parser.add_argument("--root-waypoints", action="store_true")
  args = parser.parse_args()
  out = args.out.resolve()
  out.mkdir(parents=True, exist_ok=True)
  with (out / "workflow.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = args.checkpoint.resolve()
    check_scratch(base, args.iterations, args.root_waypoints)
    if args.iterations == 20000:
      check_fast_profile(base, args.wm_lr)
    original = verify(base, args.primary.resolve(), out / "original-independent")
    record(out / "original-comparison.json", original)
    if original["confirmed"]:
      record(out / "decision.json", {"state": "confirmed", "checkpoint": str(base)})
      return
    failed = original.get("independent", original["primary"])
    if original["repeat_inconclusive"]:
      record(out / "decision.json", {"state": "needs_diagnosis",
                                    "reason": "repeated physical evaluations disagree"})
      return
    if failed["pass_rate_met"] and not failed["tracking_met"]:
      record(out / "decision.json", {
        "state": "needs_diagnosis", "reason": "common-prefix tracking error",
        "common_body_delta_cm": failed["common_body_delta_cm"],
        "pass_delta_pp": failed["pass_delta_pp"],
      })
      return
    # When the policy regresses, diagnose the windows the reference could pass.
    counts = [
      run["lost_by_reason"] if run["pass_delta_pp"] < 0 else run["candidate"]["failures"]
      for run in (failed, failed["repeat"])
    ]
    if any(c.get("path", 0) < max(1, c.get("terminated", 0), c.get("relief", 0))
           for c in counts):
      record(
        out / "decision.json", {"state": "needs_diagnosis", "observed_failures": counts,
                                "termination_terms": [r["candidate"]["termination_terms"]
                                                      for r in (failed, failed["repeat"])]}
      )
      return
    probe = out / "before-waypoints.json"
    run("probe_depth_path.py", [base, BANK, probe], out / "before-waypoints.log")
    before = read(probe)
    if before["use_root_waypoints"] or any(
      r["action_max_abs"] != 0 for r in before["offsets"]
    ):
      record(
        out / "decision.json",
        {"state": "needs_diagnosis", "reason": "offset is already observable"},
      )
      return
    if args.iterations == 20000:
      record(out / "decision.json", {
        "state": "needs_diagnosis", "reason": "20K stage complete; path drift confirmed",
        "observed_failures": counts, "next_candidate": "reference pelvis waypoints",
        "checkpoint": str(base),
      })
      return
    record(
      out / "decision.json",
      {"state": "repair_smoke", "observed_failures": counts, "evaluated_checkpoint": str(base),
       "initialization": "scratch"},
    )
    smoke = train_waypoints(out / "waypoint-smoke-scratch", 40)
    after = out / "waypoint-smoke-scratch" / "probe.json"
    run(
      "probe_depth_path.py", [smoke, BANK, after], after.with_suffix(".log")
    )
    observation = read(after)
    if not observation["use_root_waypoints"] or not all(
      r["action_mean_abs"] > 0 for r in observation["offsets"]
    ):
      raise RuntimeError("Trained Actor still ignores the waypoint perturbation")
    record(
      out / "decision.json",
      {"state": "training_waypoints", "initialization": "scratch", "iterations": args.iterations},
    )
    repaired = train_waypoints(out / f"waypoint-{args.iterations // 1000}k-scratch", args.iterations)
    evaluate(repaired, out / "waypoint-primary", "primary")
    repair = verify(repaired, out / "waypoint-primary", out / "waypoint-independent")
    repair["versus_scratch_without_waypoints"] = {
      "primary": compare(args.primary, out / "waypoint-primary", HELD_OUT, "depth.json")
    }
    if "independent" in original and "independent" in repair:
      repair["versus_scratch_without_waypoints"]["independent"] = compare(
        out / "original-independent", out / "waypoint-independent", HELD_OUT, "depth.json"
      )
    record(out / "waypoint-comparison.json", repair)
    record(
      out / "decision.json",
      {
        "state": "confirmed" if repair["confirmed"] else "needs_diagnosis",
        "checkpoint": str(repaired),
      },
    )


if __name__ == "__main__":
  main()
