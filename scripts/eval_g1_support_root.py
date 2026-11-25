"""Probe support-aware root-height reconstruction on paired rough-terrain windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from eval_g1_ankle_optimization import load_windows
from eval_g1_foothold_synthesis import SynthesisRollout, summarize
from eval_scene_flat_regression import evaluate
from probe_g1_ankle_optimization import geometry, solve_window
from probe_g1_terrain_reference import Clearance, collision_scene
from scipy.ndimage import gaussian_filter1d, median_filter

from supermimic.tasks.terrain_reference import TerrainLegReference


def adapt_window(adapter, original, height, fps=50.0, *, swing_clearance_only=False,
                 multipoint_contact=False):
  ankles, rotations, soles = geometry(adapter, original)
  floor = height(soles[..., :2])
  clearance = (soles[..., 2] - floor).min(axis=(-1, -2))
  foot_speed = np.linalg.norm(np.gradient(ankles, axis=0), axis=-1) * fps
  support = median_filter(
    (soles[..., 2].min(axis=(-1, -2)) < 0.03) & (foot_speed < 0.35), size=(3, 1)
  )
  # Preserve the nominal swing clearance while avoiding visible terrain collision.
  native_clearance = np.maximum(soles[..., 2].min(axis=(-1, -2)), 0.0)
  foot_lift = np.maximum(native_clearance - clearance, 0.0)
  if swing_clearance_only:
    # Flight and swing feet retain their mocap path unless the terrain would
    # actually intersect the sole. Stance feet retain their native clearance.
    foot_lift = np.where(support, foot_lift, np.maximum(0.005 - clearance, 0.0))
  target = ankles.copy()
  target[..., 2] += foot_lift
  weights = gaussian_filter1d(support.astype(float), 1.0, axis=0, mode="nearest")
  weight_sum = weights.sum(1)
  desired_delta = (weights * foot_lift).sum(1) / np.maximum(weight_sum, 1e-6)
  desired_delta[weight_sum < 0.05] = 0.0
  # Keep the mocap flight path and enforce an explicit per-frame height rate.
  root_delta = gaussian_filter1d(desired_delta, 2.0, mode="nearest")
  root_delta = np.clip(root_delta, -0.04, 0.04)
  for i in range(1, len(root_delta)):
    root_delta[i] = np.clip(
      root_delta[i], root_delta[i - 1] - 0.01, root_delta[i - 1] + 0.01
    )
  seed = original.copy()
  seed[:, 2] += root_delta
  contact = {}
  if multipoint_contact:
    # Select the physical front and rear sole extremes from the active MuJoCo
    # collision geometry. A point participates only when the original motion
    # planted that part of the foot; toe-off and heel-strike remain possible.
    offsets = np.empty((2, 2, 3))
    for side in range(2):
      local = (soles[0, side].reshape(-1, 3) - ankles[0, side]) @ rotations[0, side]
      x = local[:, 0]
      for point, extreme in enumerate((x.max(), x.min())):
        offsets[side, point] = local[np.abs(x - extreme) < 0.002].mean(axis=0)
    raw_points = ankles[:, :, None] + np.einsum(
      "tsij,spj->tspi", rotations, offsets
    )
    point_targets = raw_points.copy()
    point_targets[..., :2] += (target - ankles)[:, :, None, :2]
    floor = height(point_targets[..., :2])
    valid = np.isfinite(floor)
    point_targets[..., 2] = (
      np.where(valid, floor, 0.0) + np.maximum(raw_points[..., 2], 0.004)
    )
    weights = 6.0 * (support[:, :, None] & (raw_points[..., 2] < 0.012) & valid)
    contact = dict(contact_targets=point_targets, contact_offsets=offsets,
                   contact_weights=weights)
  fitted = solve_window(adapter, seed, target, rotations, support, **contact)
  if not np.isfinite(fitted).all():
    raise RuntimeError("Nonfinite support-aware reference")
  return dict(original=original, clearance_optimized=fitted), dict(
    support_samples=int(support.sum()),
    flight_frames=int((~support.any(1)).sum()),
    peak_root_target_m=float(np.abs(root_delta).max()),
    multipoint_contact=bool(multipoint_contact),
    active_contact_points=int(np.count_nonzero(contact.get("contact_weights", []))),
  )


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
  collision = Clearance(adapter)
  args.output.mkdir(parents=True, exist_ok=True)

  def height(xy):
    return terrain.height_at(torch.as_tensor(xy, dtype=torch.float32)).numpy()

  adapted = [adapt_window(adapter, row["original"], height) for row in baseline]
  sequences = {"existing": baseline, "support_root": [row[0] for row in adapted]}
  geometry_report = {}
  for arm, rows in sequences.items():
    depths = [
      [collision.contact(pose).max() for pose in row["clearance_optimized"]]
      for row in rows
    ]
    geometry_report[arm] = dict(
      penetration_over_5mm=int(
        sum(np.count_nonzero(np.array(x) > 0.005) for x in depths)
      ),
      peak_penetration_m=float(max(max(x) for x in depths)),
    )
  probes = {}
  for arm, rows in sequences.items():
    np.savez_compressed(
      args.output / f"{arm}_references.npz",
      **{
        Path(clip["path"]).stem: row["clearance_optimized"]
        for clip, row in zip(clips, rows, strict=True)
      },
    )
    probe = SynthesisRollout(clips, rows)
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
  per_motion = summarize(clips, probes)
  output = dict(
    per_motion=per_motion, geometry=geometry_report, support=[row[1] for row in adapted]
  )
  (args.output / "comparison.json").write_text(json.dumps(output, allow_nan=False))
  for arm in probes:
    metrics = [r["arms"][arm] for r in per_motion]
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
      "geometry",
      geometry_report[arm],
      flush=True,
    )


if __name__ == "__main__":
  main()
