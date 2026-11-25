"""Compare fixed-foot clearance, slope-aligned feet, and support-aware pelvis on rough ground."""

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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from supermimic.tasks.terrain_reference import TerrainLegReference


def aligned_rotation(original, points, heights):
  xy = points.reshape(-1, 2)
  z = heights.reshape(-1)
  plane = np.linalg.lstsq(np.c_[xy - xy.mean(0), np.ones(len(xy))], z, rcond=None)[0]
  normal = np.r_[-plane[:2], 1.0]
  normal /= np.linalg.norm(normal)
  axis = np.cross(original[:, 2], normal)
  angle = min(
    np.arctan2(np.linalg.norm(axis), np.dot(original[:, 2], normal)), np.deg2rad(8.0)
  )
  if np.linalg.norm(axis) < 1e-8:
    return original
  return (
    Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix() @ original
  )


def synthesize(adapter, raw, height, pelvis):
  ankles, rotations, soles = geometry(adapter, raw)
  stance = soles[..., 2].min(axis=(-1, -2)) <= 0.02
  floors = height(soles[..., :2])
  known = np.isfinite(floors)
  lift = np.where(known, floors - soles[..., 2], -np.inf).max(axis=(-1, -2))
  lift = np.where(known.any(axis=(-1, -2)), np.maximum(lift, 0.0), 0.0)
  target = ankles.copy()
  target[..., 2] += lift
  desired = rotations.copy()
  slope = []
  for foot in range(2):
    planted = np.flatnonzero(stance[:, foot])
    for run in np.split(planted, np.where(np.diff(planted) > 1)[0] + 1):
      if not len(run):
        continue
      anchor = run[len(run) // 2]
      sample_xy = soles[anchor, foot, ..., :2]
      sample_h = floors[anchor, foot]
      if not np.isfinite(sample_h).all():
        continue
      angle_before = rotations[anchor, foot]
      aligned = aligned_rotation(angle_before, sample_xy, sample_h)
      change = Rotation.from_matrix(aligned @ angle_before.T).as_rotvec()
      slope.append(float(np.rad2deg(np.linalg.norm(change))))
      desired[run, foot] = (
        Rotation.from_rotvec(change).as_matrix() @ rotations[run, foot]
      )
    # Smooth the rotation residual around touchdown and liftoff.
    residual = Rotation.from_matrix(
      desired[:, foot] @ rotations[:, foot].transpose(0, 2, 1)
    ).as_rotvec()
    residual = gaussian_filter1d(residual, 1.5, axis=0, mode="nearest")
    desired[:, foot] = Rotation.from_rotvec(residual).as_matrix() @ rotations[:, foot]
  seed = raw.copy()
  if pelvis:
    support = stance.astype(float)
    denominator = support.sum(1).clip(1)
    support_xy = (ankles[..., :2] * support[..., None]).sum(1) / denominator[:, None]
    # Transfer only a small fraction of the support shift to the pelvis.
    lateral = np.clip(0.08 * (support_xy - raw[:, :2]), -0.015, 0.015)
    lateral[~stance.any(1)] = 0.0
    vertical = 0.5 * (lift * support).sum(1) / denominator
    seed[:, :2] += gaussian_filter1d(lateral, 2.0, axis=0, mode="nearest")
    seed[:, 2] += gaussian_filter1d(vertical, 2.0, mode="nearest")
  fitted = solve_window(adapter, seed, target, desired, stance, root_xy=pelvis)
  if not np.isfinite(fitted).all():
    raise RuntimeError("Nonfinite posture reference")
  return dict(original=raw, clearance_optimized=fitted), slope


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("checkpoint", "offline", "protocol", "starts", "output"):
    parser.add_argument(name, type=Path)
  args = parser.parse_args()
  torch.set_num_threads(1)
  protocol, clips, baseline, report = load_windows(args.offline, args.protocol)
  model, ground = collision_scene(
    report["terrain_seed"], rough_scale=report["rough_scale"]
  )
  adapter = TerrainLegReference(model, ground.height_at)
  args.output.mkdir(parents=True, exist_ok=True)

  def height(xy):
    return ground.height_at(torch.as_tensor(xy, dtype=torch.float32)).numpy()

  probes, angles = {}, {}
  for arm in ("A", "B", "C"):
    trajectories = baseline
    if arm != "A":
      fitted = [
        synthesize(adapter, row["original"], height, arm == "C") for row in baseline
      ]
      trajectories = [item[0] for item in fitted]
      angles[arm] = [item[1] for item in fitted]
    np.savez_compressed(
      args.output / f"{arm}_references.npz",
      **{
        Path(c["path"]).stem: seq["clearance_optimized"]
        for c, seq in zip(clips, trajectories, strict=True)
      },
    )
    probe = SynthesisRollout(clips, trajectories)
    result = evaluate(
      args.checkpoint,
      protocol,
      args.starts,
      "cuda:0",
      rough_seed=report["terrain_seed"],
      rough_scale=report["rough_scale"],
      max_steps=max(c["horizon"] for c in clips),
      rollout_probe=probe,
    )
    probes[arm] = probe
    (args.output / f"{arm}.json").write_text(json.dumps(result, allow_nan=False))
  rows = summarize(clips, probes)
  (args.output / "comparison.json").write_text(
    json.dumps(dict(per_motion=rows, slope_angles_deg=angles), allow_nan=False)
  )
  for arm in probes:
    measures = [r["arms"][arm] for r in rows]
    matched = [m for m in measures if "peak_abs_vz" in m]
    print(
      arm,
      "matched",
      len(matched),
      "termination",
      sum(m["termination_step"] is not None for m in measures),
      "slip",
      sum(m["slip_distance_m"] for m in measures)
      / sum(m["stance_seconds"] for m in measures),
      "vz",
      np.mean([m["peak_abs_vz"] for m in matched]) if matched else None,
      "tilt",
      np.mean([m["peak_tilt_deg"] for m in matched]) if matched else None,
      flush=True,
    )


if __name__ == "__main__":
  main()
