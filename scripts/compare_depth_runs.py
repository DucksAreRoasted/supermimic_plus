"""Compare Actor runs on matched physical starts, including first-failure diagnostics."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np


def failure(row):
  if not row["survived"]:
    return "terminated"
  if row["last_root_xy_error_m"] > 0.4:
    return "path"
  if row["crossed_height_m"] < 0.02:
    return "relief"
  return "pass"


def paired_clip(reference, candidate, reference_file="concat.json"):
  with np.load(reference / "starts.npz") as a, np.load(candidate / "starts.npz") as b:
    for k in ("frames", "qpos", "qvel", "context", "full_state"):
      np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    starts = a["frames"].tolist()
  runs = [
    json.loads(p.read_text())
    for p in (reference / reference_file, candidate / "depth.json")
  ]
  for k in (
    "clip_sha256",
    "height",
    "lateral_height",
    "landscape_seed",
    "steps",
    "depth_noise_m",
    "depth_dropout",
    "perception",
    "fusion_mode",
    "seed",
    "solver_iterations",
    "solver_ls_iterations",
    "external_joint_compensation",
    "flat",
    "disjoint_windows",
  ):
    if runs[0][k] != runs[1][k]:
      raise ValueError(f"Unpaired {k} in {candidate}")
  if (
    runs[0]["external_joint_compensation"]
    or runs[0]["perception"] != "depth"
    or runs[0]["flat"]
  ):
    raise ValueError("Expected raw depth on rough ground")
  result = []
  for run in runs:
    if [r["start"] for r in run["windows_detail"]] != starts:
      raise ValueError("Physical starts do not match result rows")
    diffs = run["initial_max_abs_difference"]
    if diffs and (
      any(diffs[k] != 0 for k in ("frames", "qpos", "qvel", "context"))
      or diffs["full_state"] > 1e-6
    ):
      raise ValueError("Different initial conditions")
    for row in run["windows_detail"]:
      if (failure(row) == "pass") != row["success"] or len(
        row["body_error_trace_cm"]
      ) != row["steps"]:
        raise ValueError("Invalid physical evaluation")
    result.append(run["windows_detail"])
  return list(zip(*result, strict=True))


def compare(reference, candidate, clips, reference_file="concat.json"):
  per_clip = {
    clip: paired_clip(reference / clip, candidate / clip, reference_file)
    for clip in clips
  }
  pairs = [p for seq in per_clip.values() for p in seq]
  n = len(pairs)
  rows = [
    dict(clip=c, start=a["start"], reference=failure(a), candidate=failure(b))
    for c, seq in per_clip.items()
    for a, b in seq
  ]
  report = dict(windows=n, clips={}, transitions=rows)
  for i, name in enumerate(("reference", "candidate")):
    terminated = [p[i] for p in pairs if not p[i]["survived"]]
    report[name] = dict(
      success=sum(p[i]["success"] for p in pairs),
      survived=sum(p[i]["survived"] for p in pairs),
      common_body_error_cm=mean(
        mean(p[i]["body_error_trace_cm"][: min(a["steps"] for a in p)]) for p in pairs
      ),
      body_error_cm=mean(p[i]["mean_body_error_cm"] for p in pairs),
      root_xy_m=mean(p[i]["last_root_xy_error_m"] for p in pairs),
      reward_per_planned_step=mean(p[i]["reward_per_planned_step"] for p in pairs),
      failures=dict(Counter(failure(p[i]) for p in pairs)),
      termination_terms=dict(Counter(
        term for row in terminated for term in (row.get("termination_terms") or ["unreported"])
      )),
      mean_steps_before_termination=mean(row["steps"] for row in terminated) if terminated else None,
    )
  for clip, seq in per_clip.items():
    report["clips"][clip] = [sum(p[i]["success"] for p in seq) for i in (0, 1)]
  a, b = report["reference"], report["candidate"]
  delta = 100 * (b["success"] - a["success"]) / n
  body_delta = b["common_body_error_cm"] - a["common_body_error_cm"]
  report.update(
    pass_delta_pp=delta,
    common_body_delta_cm=body_delta,
    gained=sum(not a["success"] and b["success"] for a, b in pairs),
    lost=sum(a["success"] and not b["success"] for a, b in pairs),
    lost_by_reason=dict(Counter(failure(b) for a, b in pairs if a["success"] and not b["success"])),
    pass_rate_met=delta >= 5.0,
    tracking_met=body_delta <= 0.2,
    meets_criteria=delta >= 5.0 and body_delta <= 0.2,
  )
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("reference", type=Path)
  parser.add_argument("candidate", type=Path)
  parser.add_argument("--reference-file", default="concat.json")
  args = parser.parse_args()
  clips = sorted(
    d.name
    for d in args.reference.iterdir()
    if d.is_dir() and (d / args.reference_file).is_file()
  )
  result = compare(args.reference, args.candidate, clips, args.reference_file)
  (args.candidate / "paired.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n"
  )
  print(json.dumps({k: v for k, v in result.items() if k != "transitions"}, indent=2))
