"""Record Instinct depth and measure visibility of the next reference footfalls."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from eval_scene_flat_regression import evaluate
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  yaw_quat,
)
from PIL import Image, ImageDraw


class OcclusionRecorder:
  def __init__(self, output, clips):
    self.output = output
    self.clips = clips
    self.counts = defaultdict(lambda: np.zeros(9, dtype=int))
    self.images = defaultdict(lambda: np.zeros(4, dtype=int))
    self.writers = {}
    self.model = None

  def initialize(self, sim, future_feet):
    model = sim.sim.mj_model
    self.model = copy.copy(model)
    self.data = mujoco.MjData(self.model)
    self.static = model.body_weldid[model.geom_bodyid] == 0
    self.static_by_body_id = model.geom_bodyid == 0
    self.excluded_static_geometries = [model.geom(i).name for i in
                                       np.flatnonzero(self.static & ~self.static_by_body_id)]
    self.leg = np.zeros(model.ngeom, dtype=int)
    for geom, body in enumerate(model.geom_bodyid):
      while body:
        name = model.body(body).name
        if name.endswith(("left_hip_pitch_link", "right_hip_pitch_link")):
          self.leg[geom] = 1 if name.endswith("left_hip_pitch_link") else 2
          break
        body = model.body_parentid[body]
    if set(self.leg) != {0, 1, 2}:
      raise ValueError("Expected left and right hip subtrees in G1 geometry")
    self.model.geom_group[self.leg > 0] = 5
    self.landings = []
    start = 0
    for end in future_feet.motion.motion_ends.cpu().tolist():
      z = future_feet.xyz[start:end, :, 2].cpu().numpy()
      # A reference-only stance proxy. Index each low-height transition by foot.
      stance = z <= np.quantile(z, .10, axis=0) + .035
      touchdown = stance[1:] & ~stance[:-1]
      self.landings.append((start, stance, [np.flatnonzero(side) + 1 for side in touchdown.T]))
      start = end

  def record(self, sim, camera, motion, future_feet, ground, patch, root, active):
    if self.model is None:
      self.initialize(sim, future_feet)
    indices = (active & camera.refresh_mask).nonzero().flatten().tolist()
    if not indices:
      return
    now = motion.time_steps
    reference = motion.motion
    next_frames = np.repeat(now.cpu().numpy()[:, None], 2, axis=1)
    phases = {}
    for i in indices:
      start, stance, touchdowns = self.landings[i]
      frame = int(now[i]) - start
      phases[i] = []
      for side, events in enumerate(touchdowns):
        following = np.searchsorted(events, frame, side="right")
        if following == len(events):
          phases[i].append(None)
          continue
        landing = events[following]
        next_frames[i, side] = start + landing
        if stance[frame, side]:
          phase = "stance"
        elif landing - frame <= 3:
          phase = "pre_touchdown"
        elif frame >= 3 and stance[frame - 3:frame, side].any():
          phase = "lift_off"
        else:
          phase = "mid_swing"
        phases[i].append(phase)

    device = root.device
    next_frames = torch.as_tensor(next_frames, device=device)
    side = torch.arange(2, device=device)
    targets = future_feet.xyz[next_frames, side]
    ref_yaw = yaw_quat(reference.root_quat_w[now])[:, None, :].expand(-1, 2, -1).contiguous()
    centers = quat_apply_inverse(ref_yaw, targets - reference.root_pos_w[now, None])
    offsets = root.new_tensor([[0, 0, 0], [.08, 0, 0], [-.08, 0, 0], [0, .04, 0], [0, -.04, 0]])
    foot_yaw = yaw_quat(future_feet.quat[next_frames, side].reshape(-1, 4)).reshape_as(ref_yaw)
    offsets = quat_apply(foot_yaw[:, :, None].expand(-1, -1, 5, -1), offsets.expand(len(root), 2, -1, -1))
    offsets = quat_apply_inverse(ref_yaw[:, :, None].expand(-1, -1, 5, -1), offsets)
    local = (centers[:, :, None] + offsets).flatten(1, 2)
    known = patch.sample(root, local[..., :2])[..., 1]
    pelvis_yaw = yaw_quat(root[:, 3:7])[:, None].expand(-1, 10, -1).contiguous()
    points = quat_apply(pelvis_yaw, local) + root[:, None, :3]
    points[..., 2] = ground.height_at(points[..., :2])
    cam = camera.data
    relative = quat_apply_inverse(
      cam.quat_w_world[:, None].expand(-1, 10, -1).contiguous(), points - cam.pos_w[:, None]
    )
    depth = cam.output["distance_to_image_plane"].reshape(-1, 27, 48).cpu().numpy()
    hits = camera._read_backend_hit_geom_ids().cpu().numpy()
    qpos = sim.sim.data.qpos.cpu().numpy()
    rotation = matrix_from_quat(cam.quat_w_world).cpu().numpy()
    for i in indices:
      starts = camera.ray_starts[i].cpu().numpy()
      dirs = camera.ray_directions[i].cpu().numpy()
      origin = starts[0]
      if not np.allclose(starts, origin):
        raise ValueError("Expected one ray origin for the pinhole camera")
      target = relative[i].cpu().numpy() - origin
      pixel, in_view = self.project(target, dirs)
      raw = depth[i].reshape(-1)
      labels = np.where(hits[i] >= 0, self.leg[hits[i].clip(min=0)], 0)
      without, unmasked_hits = self.without_legs(qpos[i], cam.pos_w[i].cpu().numpy() + rotation[i] @ origin,
                                  dirs, rotation[i])
      target_depth = target[:, 0] + origin[0]
      candidate = in_view & (target_depth >= .05) & (target_depth <= 2)
      before = raw[pixel] < target_depth - .045
      ground_after = (unmasked_hits >= 0) & self.static[unmasked_hits.clip(min=0)]
      uncovered = ground_after[pixel] & (abs(without[pixel] - target_depth) <= .075)
      leg_block = candidate & before & (labels[pixel] > 0) & uncovered
      ground_hit = (hits[i] >= 0) & self.static[hits[i].clip(min=0)]
      visible = candidate & ground_hit[pixel] & (abs(raw[pixel] - target_depth) <= .075)
      map_valid = known[i].cpu().numpy().astype(bool)
      for side_id, phase in enumerate(phases[i]):
        if phase is None:
          continue
        part = slice(side_id * 5, side_id * 5 + 5)
        self.counts[(self.clips[i], ("left", "right")[side_id], phase)] += [
          5, visible[part].sum(), (leg_block & (labels[pixel] == 1))[part].sum(),
          (leg_block & (labels[pixel] == 2))[part].sum(), (~in_view)[part].sum(),
          map_valid[part].sum(), (candidate & ~visible & ~leg_block)[part].sum(),
          (in_view & ~candidate)[part].sum(), (leg_block & map_valid)[part].sum(),
        ]
      self.images[self.clips[i]] += [1, len(raw), (labels == 1).sum(), (labels == 2).sum()]
      self.write_frame(i, raw, without, labels, pixel, in_view, phases[i], sim.step_dt)

  @staticmethod
  def project(target, dirs):
    uv = dirs[:, 1:] / dirs[:, :1]
    points = target[:, 1:] / np.maximum(target[:, :1], 1e-6)
    corners = uv[[0, 47, -1, -48]]
    edges = np.roll(corners, -1, axis=0) - corners
    rel = points[:, None] - corners
    cross = edges[:, 0] * rel[..., 1] - edges[:, 1] * rel[..., 0]
    inside = ((cross >= 0).all(1) | (cross <= 0).all(1)) & (target[:, 0] > 0)
    pixel = np.square(points[:, None] - uv).sum(-1).argmin(1)
    return pixel, inside

  def without_legs(self, qpos, origin, dirs, rotation):
    self.data.qpos[:] = qpos
    mujoco.mj_forward(self.model, self.data)
    dist = np.empty(len(dirs), np.float64)
    hit = np.empty(len(dirs), np.int32)
    directions = np.asarray(dirs @ rotation.T, dtype=np.float64)
    origin = origin.astype(np.float64)
    groups = np.array([1, 0, 1, 0, 0, 0], np.uint8)
    mujoco.mj_multiRay(self.model, self.data, origin, directions.flatten(),
                       groups, 1, -1, hit, dist, None, len(dirs), 5.0)
    for i in np.flatnonzero((dist >= 0) & (dist <= .05)):
      walked = 0.0
      tag = np.array([-1], dtype=np.int32)
      for _ in range(25):
        step = mujoco.mj_ray(self.model, self.data, origin + walked * directions[i],
                             directions[i], groups, 1, -1, tag)
        if step < 0 or walked + step > .05:
          dist[i], hit[i] = (-1, -1) if step < 0 else (walked + step, tag[0])
          break
        walked += step + 1e-4
    return dist * dirs[:, 0], hit

  def write_frame(self, i, raw, without, labels, pixels, in_view, phases, dt):
    clip = self.clips[i]
    if clip not in self.writers:
      self.writers[clip] = imageio.get_writer(self.output / f"{clip}.mp4", fps=round(1 / dt), macro_block_size=2)
    panels = []
    for distances, label in ((raw, labels), (without, np.zeros_like(labels))):
      value = np.where(distances > 0, 255 * (1 - distances.clip(0, 2) / 2), 0).astype(np.uint8)
      rgb = np.repeat(value[:, None], 3, axis=1)
      rgb[label == 1] = [230, 75, 60]
      rgb[label == 2] = [243, 168, 57]
      panel = Image.fromarray(rgb.reshape(27, 48, 3)).resize((384, 216), Image.Resampling.NEAREST)
      marks = ImageDraw.Draw(panel)
      for point, (pixel, visible) in enumerate(zip(pixels, in_view, strict=True)):
        if visible and phases[point // 5] is not None:
          y, x = divmod(int(pixel), 48)
          marks.rectangle((8*x+1, 8*y+1, 8*x+6, 8*y+6), outline=("cyan", "magenta")[point // 5])
      panels.append(panel)
    frame = Image.new("RGB", (768, 256), "#132333")
    frame.paste(panels[0], (0, 20))
    frame.paste(panels[1], (384, 20))
    text = ImageDraw.Draw(frame)
    text.text((8, 4), "Instinct measured depth | red/orange: left/right leg", fill="white")
    text.text((392, 4), "Same pose and rays | leg geometry excluded", fill="white")
    text.text((8, 238), f"{clip}  |  cyan L: {phases[0]}  |  magenta R: {phases[1]}", fill="white")
    self.writers[clip].append_data(np.asarray(frame))

  def summary(self):
    fields = ("samples", "visible", "blocked_left", "blocked_right", "out_of_view", "map_valid", "other_in_range", "out_of_range", "blocked_map_valid")
    return {
      "landing_counts": [dict(motion=clip, foot=side, phase=phase, **dict(zip(fields, values.tolist(), strict=True)))
                         for (clip, side, phase), values in sorted(self.counts.items())],
      "fixed_geometries_excluded_by_terrain_only": self.excluded_static_geometries,
      "image_counts": {key: dict(zip(("frames", "pixels", "left_leg", "right_leg"), value.tolist(), strict=True))
                       for key, value in self.images.items()},
    }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for arg in ("checkpoint", "protocol", "output"):
    parser.add_argument(arg, type=Path)
  parser.add_argument("--clips", nargs="+", required=True, help="Motion filenames without .npz")
  parser.add_argument("--steps", type=int, default=180)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--rough-seed", type=int)
  parser.add_argument("--rough-scale", type=float, default=.25)
  args = parser.parse_args()
  protocol = json.loads(args.protocol.read_text())
  protocol["clips"] = [c for c in protocol["clips"] if Path(c["path"]).stem in args.clips]
  clips = [Path(c["path"]).stem for c in protocol["clips"]]
  if set(clips) != set(args.clips) or not (args.steps > 0 and 0 <= args.start < min(c["frames"] for c in protocol["clips"])):
    parser.error("Choose valid clip names, start and step count")
  for clip in protocol["clips"]:
    clip["local_start"] = args.start
  args.output.mkdir(parents=True)
  recorder = OcclusionRecorder(args.output, clips)
  try:
    result = evaluate(args.checkpoint, protocol, args.output / "starts.npz", "cuda:0",
                      rough_seed=args.rough_seed, rough_scale=args.rough_scale, save_starts=True,
                      camera_diagnostic=recorder, max_steps=args.steps)
  finally:
    for writer in recorder.writers.values():
      writer.close()
  report = recorder.summary()
  report["metadata"] = dict(checkpoint=str(args.checkpoint), rough_seed=args.rough_seed,
                            rough_scale=args.rough_scale, start=args.start, max_steps=args.steps,
                            evaluated_steps={r["motion"]: r["steps"] for r in result["per_motion"]})
  (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  main()
