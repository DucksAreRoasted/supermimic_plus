"""Paired, short-horizon terrain-conformal reference synthesis on G1 failure windows."""

import argparse
import json
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import torch
from eval_g1_ankle_optimization import AnkleRollout, load_windows
from eval_scene_flat_regression import evaluate
from probe_g1_ankle_optimization import geometry, solve_window
from probe_g1_terrain_reference import collision_scene
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial.transform import Rotation

from supermimic.tasks.terrain_reference import TerrainLegReference

PLAN_STEP = (
  6  # All arms execute identical raw-reference actions through the first six steps.
)


class CameraSnapshot:
  """Capture a causal world-aligned query interface to the evaluator's measured patch."""

  def __init__(self):
    self.step = 0

  def record(self, sim, camera, motion, feet, ground, patch, aux, active):
    if self.step == PLAN_STEP:
      if not active.all():
        raise RuntimeError("A rollout terminated before the common planning frame")
      self.image = patch.image.cpu().numpy().copy()
      self.origin = patch.origin.cpu().numpy().copy()
      self.xy = patch.crop_xy.cpu().numpy().copy()
    self.step += 1

  def lookup(self, index, points):
    origin = self.origin[index]
    yaw = Rotation.from_quat(origin[2:], scalar_first=True).as_matrix()[:2, :2]
    local = (points.reshape(-1, 2) - origin[:2]) @ yaw
    grid = (local - self.xy[0, 0]) / (self.xy[0, 1, 0] - self.xy[0, 0, 0])
    height, valid = [
      map_coordinates(channel, grid[:, ::-1].T, order=1, mode="constant", cval=0.0)
      for channel in self.image[index]
    ]
    result = np.full_like(height, np.nan)
    np.divide(height, 10 * valid, out=result, where=valid > 0.25)
    return result.reshape(points.shape[:-1])


def synthesize(adapter, raw, height):
  ankles, rotations, soles = geometry(adapter, raw)
  stance = soles[..., 2].min(axis=(-1, -2)) <= 0.02
  target = ankles.copy()
  events = []
  axis = np.arange(-2, 3) * 0.025
  offsets = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
  for foot in range(2):
    planted = np.flatnonzero(stance[:, foot])
    for run in np.split(planted, np.where(np.diff(planted) > 1)[0] + 1):
      if not len(run):
        continue
      anchor = np.median(ankles[run, foot, :2], axis=0)
      locations = anchor + offsets
      shift = locations[:, None] - ankles[run, foot, :2]
      points = soles[None, run, foot, ..., :2] + shift[:, :, None, None, :]
      floors = height(points)
      # Require the entire swept sole footprint to be observed. Large ankle
      # excursions (e.g. imperfect stance segmentation) stay on the original path.
      feasible = np.isfinite(floors).all(axis=(1, 2, 3)) & (
        np.linalg.norm(shift, axis=-1).max(1) <= 0.09
      )
      event = dict(
        foot=foot, start=int(run[0]), end=int(run[-1]), candidates=int(feasible.sum())
      )
      if feasible.any():
        ids = np.flatnonzero(feasible)
        relief = np.ptp(floors[ids], axis=(2, 3)).mean(1)
        score = relief + 0.12 * np.linalg.norm(offsets[ids], axis=-1)
        best = ids[score.argmin()]
        target[run, foot, :2] = locations[best]
        event["offset_m"] = offsets[best].tolist()
      events.append(event)
    # Interpolate displacement on swing frames, with zero endpoint slope.
    keys = np.unique(np.r_[0, planted, len(raw) - 1])
    for a, b in zip(keys[:-1], keys[1:], strict=True):
      if b <= a + 1:
        continue
      t = (np.arange(a + 1, b) - a)[:, None] / (b - a)
      w = t**2 * (3 - 2 * t)
      d = target[[a, b], foot, :2] - ankles[[a, b], foot, :2]
      target[a + 1 : b, foot, :2] += (1 - w) * d[0] + w * d[1]
  xy_delta = target[..., :2] - ankles[..., :2]
  floor = height(soles[..., :2] + xy_delta[:, :, None, None, :])
  # Preserve nominal swing clearance over observed terrain. Unknown sole samples
  # contribute no height claim; an entirely unseen foot retains nominal height.
  known = np.isfinite(floor)
  needed = np.where(known, floor - soles[..., 2], -np.inf).max(axis=(-1, -2))
  nominal_clearance = np.maximum(0.0, soles[..., 2].min(axis=(-1, -2)))
  lift = np.where(known.any(axis=(-1, -2)), needed + nominal_clearance, 0.0)
  target[..., 2] += lift
  weights = stance[..., None].astype(float)
  support_shift = ((target - ankles) * weights).sum(1) / weights.sum(1).clip(1)
  seed = raw.copy()
  seed[:, :3] += 0.5 * gaussian_filter1d(support_shift, 2.0, axis=0, mode="nearest")
  fitted = solve_window(adapter, seed, target, rotations, stance)
  if not np.isfinite(fitted).all():
    raise RuntimeError("Nonfinite synthesized pose")
  return dict(original=raw, clearance_optimized=fitted), events


class SynthesisRollout(AnkleRollout):
  def bind(self, sim, command, feet):
    super().bind(sim, command, feet)
    self.site_ids = self.robot.find_sites(
      ["left_foot", "right_foot"], preserve_order=True
    )[0]
    self.contact_to_site = [
      0 if "left" in name else 1 for name in self.contact.primary_names
    ]
    if sorted(self.contact_to_site) != [0, 1]:
      raise ValueError("Expected one contact sensor per foot")
    self.step = 0

  def adapt(self, obs, representation, motion_times, aux):
    # Use nominal feet everywhere outside the fitted reference, including warmup.
    # This keeps the oracle checkpoint's built-in terrain labels out of the camera arm.
    obs["future_feet"] = self.nominal_feet(motion_times)
    if self.step >= PLAN_STEP:
      super().adapt(obs, representation, motion_times, aux)

  def record(self, step, active):
    self.step = step
    super().record(step, active)
    xy = self.robot.data.site_pos_w[:, self.site_ids, :2].cpu().numpy()
    for i, trace in enumerate(self.trace):
      if trace and trace[-1]["step"] == step:
        trace[-1]["sole_xy"] = xy[i, self.contact_to_site].tolist()


def summarize(clips, probes):
  rows = []
  for i, clip in enumerate(clips):
    center = clip["center_step"]
    row = dict(motion=Path(clip["path"]).stem, arms={})
    # Choose one physical foot with a touchdown in all three arms.
    events = {k: p.events(i) for k, p in probes.items()}
    choices = []
    for foot in range(2):
      steps = [
        [s for s, f in arm_events if f == foot and s > PLAN_STEP]
        for arm_events in events.values()
      ]
      for times in product(*steps):
        spread = max(times) - min(times)
        if spread <= 5:  # Same foot within 100 ms, after the common warmup.
          choices.append((sum(abs(s - center) for s in times), spread, foot, times))
    matched = min(choices) if choices else None
    if matched:
      _, _, row["contact_foot"], times = matched
      match = dict(zip(probes, times, strict=True))
      length = min(11, *(len(p.trace[i]) - match[k] for k, p in probes.items()))
      row["post_contact_samples"] = length
    for k, p in probes.items():
      trace = p.trace[i]
      xy = np.array([x["sole_xy"] for x in trace])
      contacts = np.array([x["contact"] for x in trace], dtype=bool)
      stance = contacts[1:] & contacts[:-1]
      dist = np.linalg.norm(np.diff(xy, axis=0), axis=-1)[stance]
      metrics = dict(
        termination_step=p.termination[i],
        events=events[k],
        slip_distance_m=float(dist.sum()),
        stance_seconds=float(stance.sum() * p.dt),
      )
      if matched:
        post = trace[match[k] : match[k] + length]
        metrics.update(
          contact_step=match[k],
          peak_abs_vz=max(abs(x["pelvis_vz_m_s"]) for x in post),
          peak_tilt_deg=max(x["torso_tilt_deg"] for x in post),
        )
      row["arms"][k] = metrics
    rows.append(row)
  return rows


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
  snapshot = CameraSnapshot()
  probes, selected = {}, {}

  def true_height(xy):
    return ground.height_at(torch.as_tensor(xy, dtype=torch.float32)).numpy()

  for arm in ("fixed", "oracle", "camera"):
    trajectories = baseline
    if arm != "fixed":
      trajectories, selected[arm] = [], []
      for i, sequence in enumerate(baseline):
        height = partial(snapshot.lookup, i) if arm == "camera" else true_height
        fit, choices = synthesize(adapter, sequence["original"], height)
        trajectories.append(fit)
        selected[arm].append(choices)
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
      camera_diagnostic=snapshot if arm == "fixed" else None,
    )
    probes[arm] = probe
    (args.output / f"{arm}.json").write_text(json.dumps(result, allow_nan=False))
  np.savez_compressed(
    args.output / "camera_snapshot.npz",
    image=snapshot.image,
    origin=snapshot.origin,
    xy=snapshot.xy,
  )
  rows = summarize(clips, probes)
  output = dict(
    plan_step=PLAN_STEP,
    selected=selected,
    per_motion=rows,
    traces={k: p.trace for k, p in probes.items()},
  )
  (args.output / "comparison.json").write_text(json.dumps(output, allow_nan=False))
  for arm in probes:
    ms = [r["arms"][arm] for r in rows]
    matched = [m for m in ms if "peak_abs_vz" in m]
    print(
      arm,
      "matched",
      len(matched),
      "terminated",
      sum(m["termination_step"] is not None for m in ms),
      "slip m/s",
      sum(m["slip_distance_m"] for m in ms) / sum(m["stance_seconds"] for m in ms),
      "peaks",
      [
        np.mean([m[key] for m in matched]) if matched else None
        for key in ("peak_abs_vz", "peak_tilt_deg")
      ],
    )


if __name__ == "__main__":
  main()
