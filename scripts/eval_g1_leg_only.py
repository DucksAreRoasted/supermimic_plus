"""Paired rollout of clearance IK with fixed or adjustable pelvis height."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from eval_g1_ankle_optimization import load_windows
from eval_g1_foothold_synthesis import SynthesisRollout, summarize
from eval_scene_flat_regression import evaluate
from probe_g1_ankle_optimization import geometry, solve_window
from probe_g1_terrain_reference import collision_scene

from supermimic.tasks.terrain_reference import TerrainLegReference


def leg_only(adapter, sequence, height):
  raw = sequence["original"]
  ankles, rotations, soles = geometry(adapter, raw)
  support = soles[..., 2].min(axis=(-1, -2)) <= 0.02
  floor = height(soles[..., :2])
  clearance = (soles[..., 2] - floor).min(axis=(-1, -2))
  nominal_lift = sequence["ankle_only_target"][..., 2] - ankles[..., 2]
  target = ankles.copy()
  target[..., 2] += np.maximum(nominal_lift, -clearance)
  corrected = solve_window(adapter, raw, target, rotations, support, root_z=False)
  np.testing.assert_array_equal(corrected[:, :7], raw[:, :7])
  return dict(original=raw, clearance_optimized=corrected)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for key in ("checkpoint", "offline", "protocol", "starts", "output"):
    p.add_argument(key, type=Path)
  args = p.parse_args()
  torch.set_num_threads(1)
  protocol, clips, baseline, report = load_windows(args.offline, args.protocol)
  model, terrain = collision_scene(
    report["terrain_seed"], rough_scale=report["rough_scale"]
  )
  adapter = TerrainLegReference(model, terrain.height_at)
  args.output.mkdir(parents=True, exist_ok=True)

  def height(xy):
    return terrain.height_at(torch.as_tensor(xy, dtype=torch.float32)).numpy()

  probes = {}
  for arm in ("pelvis_z", "legs_only"):
    sequences = (
      baseline
      if arm == "pelvis_z"
      else [leg_only(adapter, row, height) for row in baseline]
    )
    np.savez_compressed(
      args.output / f"{arm}_references.npz",
      **{
        Path(clip["path"]).stem: seq["clearance_optimized"]
        for clip, seq in zip(clips, sequences, strict=True)
      },
    )
    probe = SynthesisRollout(clips, sequences)
    result = evaluate(
      args.checkpoint,
      protocol,
      args.starts,
      "cuda:0",
      rough_seed=report["terrain_seed"],
      rough_scale=report["rough_scale"],
      max_steps=max(clip["horizon"] for clip in clips),
      rollout_probe=probe,
    )
    probes[arm] = probe
    (args.output / f"{arm}.json").write_text(json.dumps(result, allow_nan=False))
  rows = summarize(clips, probes)
  (args.output / "comparison.json").write_text(
    json.dumps(dict(per_motion=rows), allow_nan=False)
  )
  for arm in probes:
    metrics = [r["arms"][arm] for r in rows]
    matched = [m for m in metrics if "peak_abs_vz" in m]
    print(
      arm,
      "matched",
      len(matched),
      "terminated",
      sum(m["termination_step"] is not None for m in metrics),
      "slip",
      sum(m["slip_distance_m"] for m in metrics)
      / sum(m["stance_seconds"] for m in metrics),
      "vz",
      np.mean([m["peak_abs_vz"] for m in matched]) if matched else None,
      "tilt",
      np.mean([m["peak_tilt_deg"] for m in matched]) if matched else None,
      flush=True,
    )


if __name__ == "__main__":
  main()
