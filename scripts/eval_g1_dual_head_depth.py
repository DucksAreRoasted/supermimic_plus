"""Evaluate the final online checkpoint on the previous depth pilot's paired starts."""

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean

from compare_depth_runs import compare
from run_foot_depth_pilot import ROOT, evaluate
from validate_depth_mimic import CASES


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("out", type=Path)
  parser.add_argument("--paired-starts", type=Path)
  parser.add_argument("--panel", choices=("primary", "independent"), default="primary")
  parser.add_argument(
    "--baseline-adapter", type=Path, help="Build the independent comparison reference"
  )
  args = parser.parse_args()
  building_reference = args.baseline_adapter is not None
  reuse_starts = not building_reference or args.paired_starts is not None
  if building_reference and args.baseline_adapter.stem != "concat":
    parser.error("Use the original concat adapter to build a paired reference")
  if args.paired_starts is None:
    args.paired_starts = ROOT / (
      "outputs/foot-depth-pilot/holdout"
      if args.panel == "primary"
      else "outputs/dual-head-depth/independent-reference"
    )
  checkpoint = args.checkpoint.resolve()
  manifest = json.loads((checkpoint / "run_manifest.json").read_text())
  if not building_reference and (
    manifest["iteration"] != manifest["max_iterations"]
    or manifest["runner_cfg"]["depth_fusion"] != "concat"
  ):
    raise ValueError("Expected a completed online depth checkpoint")
  rows = []
  clips = {}
  for suite, count in (("confirmation", 12), ("prospective", 16)):
    for case in CASES[suite]:
      clip = case[0]
      if clip == "walk1_subject1":
        continue
      directory = args.out.resolve() / clip
      directory.mkdir(parents=True, exist_ok=True)
      common = args.paired_starts / clip / "starts.npz"
      snapshot = directory / "starts.npz"
      frame_range = (4000, 11800)
      if args.panel == "independent":
        clip, seed, height, lateral, layout = case
        case = (clip, seed + 10000, height, lateral, layout + 10000)
        count, frame_range = 8, (0, 3900)
      if reuse_starts:
        if not snapshot.exists():
          shutil.copyfile(common, snapshot)
        elif snapshot.read_bytes() != common.read_bytes():
          raise ValueError(f"Initial states differ for {clip}")
      evaluate(
        checkpoint,
        case,
        directory,
        count,
        300,
        stage="holdout",
        weights=args.baseline_adapter,
        frame_range=frame_range,
      )
      run = json.loads(
        (
          directory / ("concat.json" if building_reference else "depth.json")
        ).read_text()
      )
      if (
        run["external_joint_compensation"]
        or run["fusion_mode"] != "concat"
        or run["windows"] != count
      ):
        raise ValueError(f"Unexpected physical evaluation for {clip}")
      clips[clip] = dict(
        passed=run["successes"], windows=count, body_error_cm=run["mean_body_error_cm"]
      )
      rows.extend(run["windows_detail"])
  result = dict(
    checkpoint=str(checkpoint),
    clips=clips,
    windows=len(rows),
    passed=sum(r["success"] for r in rows),
    survived=sum(r["survived"] for r in rows),
    body_error_cm=mean(r["mean_body_error_cm"] for r in rows),
    last_root_xy_error_m=mean(r["last_root_xy_error_m"] for r in rows),
  )
  if not building_reference:
    paired = compare(args.paired_starts, args.out, list(clips))
    (args.out / "paired.json").write_text(
      json.dumps(paired, indent=2, allow_nan=False) + "\n"
    )
    result["paired"] = {k: v for k, v in paired.items() if k != "transitions"}
  (args.out / "summary.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n"
  )
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
