"""Compare raw, pelvis-only, and leg-retargeted references in the actual MuJoCo collision scene."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.scene.scene import Scene
from scipy.spatial.transform import Rotation

from supermimic.tasks.terrain_reference import TerrainLegReference
from supermimic.tasks.walk_terrain import TerrainObservation
from train_g1_dual_head_depth import CLIPS, ROOT, load_env_cfg, protocol_clips, scratch_config


def collision_scene(terrain_seed, *, flat=False):
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  scratch_config(20000, 1024, 42, True, 4096).configure_terrain(cfg)
  cfg.scene.num_envs = 1
  cfg.scene.spec_fn = replace(cfg.scene.spec_fn, landscape_seed=terrain_seed)
  if flat:
    cfg.scene.spec_fn = replace(cfg.scene.spec_fn, height=0.0, lateral_height=0.0)
  return Scene(cfg.scene, device="cpu").compile(), TerrainObservation(cfg.scene.spec_fn)


def poses_from_clip(rows, frames, model):
  addr = [model.jnt_qposadr[model.joint("robot/" + str(s)).id] for s in rows["joint_names"]]
  poses = np.tile(model.qpos0, (len(frames), 1))
  poses[:, :3] = rows["root_pos_w"][frames]
  poses[:, 3:7] = rows["root_quat_w"][frames]
  poses[:, addr] = rows["joint_pos"][frames]
  return poses


class Clearance:
  def __init__(self, adapter):
    self.adapter = adapter
    m = adapter.model
    self.data = mujoco.MjData(m)
    floor = np.isin(m.geom_type, [int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_PLANE)])
    self.ground = {i for i, flag in enumerate(floor) if flag}
    self.groups = [set(ids.tolist()) for ids in adapter.geom]

  def contact(self, pose):
    m, d = self.adapter.model, self.data
    d.qpos[:] = pose
    mujoco.mj_fwdPosition(m, d)
    depth = np.zeros(2)
    for c in d.contact:
      a, b = c.geom
      if a in self.ground or b in self.ground:
        other = b if a in self.ground else a
        for side, group in enumerate(self.groups):
          if other in group:
            depth[side] = max(depth[side], -c.dist)
    return depth

  def signed(self, pose, plane_clearance):
    """Vertical translation to first physical contact, signed positive for a swing foot."""
    q = pose.copy()
    values = []
    for side in range(2):
      lo = -plane_clearance[side] - 0.025
      hi = -plane_clearance[side] + 0.25
      q[2] = pose[2] + lo
      assert self.contact(q)[side] > 0
      q[2] = pose[2] + hi
      assert self.contact(q)[side] == 0
      for _ in range(11):
        mid = 0.5 * (lo + hi)
        q[2] = pose[2] + mid
        if self.contact(q)[side] > 0:
          lo = mid
        else:
          hi = mid
      values.append(-0.5 * (lo + hi))
    return np.array(values)


def describe(x):
  x = np.asarray(x)
  return dict(mean=float(x.mean()), p95=float(np.quantile(x, .95)), max=float(x.max()))


def audit_raw_clearance(protocol, seed, count):
  """Count native MuJoCo foot contacts on evenly spaced poses in each original motion."""
  model, terrain = collision_scene(seed)
  flat_model, flat_terrain = collision_scene(seed, flat=True)
  rough = Clearance(TerrainLegReference(model, terrain.height_at))
  flat = Clearance(TerrainLegReference(flat_model, flat_terrain.height_at))
  rows = []
  totals = np.zeros(3, dtype=int)
  for path in protocol_clips(protocol):
    with np.load(path) as clip:
      if not 1 <= count <= len(clip["joint_pos"]):
        raise ValueError("Sample count must fit each motion")
      frames = np.linspace(0, len(clip["joint_pos"])-1, count).round().astype(int)
      poses = poses_from_clip(clip, frames, model)
      flat_poses = poses_from_clip(clip, frames, flat_model)
    lifted = poses.copy()
    lifted[:, 2] += terrain.height_at(torch.as_tensor(poses[:, :2], dtype=torch.float32)).numpy()
    depths = np.array([[flat.contact(a).max(), rough.contact(b).max(), rough.contact(c).max()]
                       for a,b,c in zip(flat_poses, poses, lifted, strict=True)])
    totals += (depths > .02).sum(axis=0)
    rows.append(dict(motion=path.stem, frames=frames.tolist(), max_foot_contact_penetration_m=depths.tolist()))
  return dict(terrain_seed=seed, clips=len(rows), frames_per_clip=count,
              reference_frames_with_any_foot_penetration_over_2cm=totals.tolist(),
              order=["flat", "raw_on_rough", "root_lifted_every_frame"], records=rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument("--terrain-seed", type=int, default=701)
  parser.add_argument("--segment-seed", type=int, default=9271)
  parser.add_argument("--smoothing-frames", type=float, default=2.0)
  parser.add_argument("--frames", type=int, default=96)
  parser.add_argument("--clips", nargs="+", choices=CLIPS, default=CLIPS)
  parser.add_argument("--audit-raw-lafan40", type=Path, metavar="PROTOCOL",
                      help="Measure all 40 raw clips on flat and rough collision meshes, with --frames poses per clip")
  args = parser.parse_args()
  torch.set_num_threads(1)
  if args.audit_raw_lafan40:
    report = audit_raw_clearance(args.audit_raw_lafan40, args.terrain_seed, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False) + "\n")
    print(report["reference_frames_with_any_foot_penetration_over_2cm"])
    return
  m, terrain = collision_scene(args.terrain_seed)
  adapter = TerrainLegReference(m, terrain.height_at)
  meter = Clearance(adapter)
  rng = np.random.default_rng(args.segment_seed)
  records = []
  raw_speed, adapted_speed = [], []
  velocity_changes, angle_changes, paths, orientation, hashes = [], [], [], [], {}
  masks = np.ones(m.nq, dtype=np.bool_)
  masks[adapter.qaddr.ravel()] = False
  masks[2] = False
  for clip in args.clips:
    path = ROOT / "data/modelscope/LAFAN1_Retargeting_Dataset/npz" / (clip + ".npz")
    hashes[clip] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as z:
      start = rng.integers(0, len(z["root_pos_w"]) - args.frames)
      frames = np.arange(start, start + args.frames)
      poses = poses_from_clip(z, frames, m)
      fps = float(z["fps"].item())
    root_lifts = terrain.height_at(torch.tensor(poses[:, :2], dtype=torch.float32)).numpy()
    adapted = adapter.clip(poses, args.smoothing_frames)
    for sequence, storage in ((poses, raw_speed), (adapted, adapted_speed)):
      storage.extend(np.abs(np.diff(sequence[:, adapter.qaddr.ravel()], axis=0)).ravel() * fps)
    assert np.isfinite(adapted).all() and np.array_equal(poses[:, masks], adapted[:, masks])
    joint_delta = adapted[:, adapter.qaddr.ravel()] - poses[:, adapter.qaddr.ravel()]
    velocity_changes.extend(np.abs(np.diff(joint_delta, axis=0)).ravel() * fps)
    angle_changes.extend(np.abs(joint_delta).ravel())
    for i, raw in enumerate(poses):
      root_only = raw.copy()
      root_only[2] += root_lifts[i]
      estimates, base, base_ankle, base_rot = [], None, None, None
      for q in (raw, root_only, adapted[i]):
        adapter.data.qpos[:] = q
        mujoco.mj_kinematics(m, adapter.data)
        ankle = adapter.data.xpos[adapter.ankles].copy()
        rot = adapter.data.xmat[adapter.ankles].reshape(2, 3, 3).copy()
        flat = adapter.sole_points()[..., 2].min(axis=(-1, -2))
        if base is None:
          base, base_ankle, base_rot = flat, ankle, rot
        estimates.append((meter.contact(q).copy(), meter.signed(q, flat)))
      paths.extend(np.linalg.norm(ankle[:, :2] - base_ankle[:, :2], axis=-1) * 100)
      orientation.extend(np.rad2deg(Rotation.from_matrix(rot @ base_rot.transpose(0, 2, 1)).magnitude()))
      records.append((clip, int(frames[i]), base.tolist(), np.array(estimates).tolist()))
    print(f"Measured {clip} frames {frames[0]} through {frames[-1]}", flush=True)
  flat = np.array([v[2] for v in records])
  data = np.array([v[3] for v in records])
  metrics = {}
  for i, name in enumerate(("raw", "pelvis_only", "legs")):
    depth, clearance = data[:, i, 0], data[:, i, 1]
    error = np.abs(clearance - flat) * 100
    stance, swing = flat <= 0.02, flat >= 0.04
    metrics[name] = dict(frames_with_over_2cm_penetration=int((depth.max(axis=-1) > .02).sum()),
                         stance_samples=int(stance.sum()), swing_samples=int(swing.sum()),
                         stance_clearance_mae_cm=float(error[stance].mean()) if stance.any() else None,
                         swing_clearance_mae_cm=float(error[swing].mean()) if swing.any() else None,
                         stance_hover_over_2cm=int((stance & (clearance > flat+.02)).sum()))
  report = dict(terrain_seed=args.terrain_seed, segment_seed=args.segment_seed,
                smoothing_frames=args.smoothing_frames, nominal_speed_rad_s=describe(raw_speed),
                adapted_speed_rad_s=describe(adapted_speed), frames=len(records), clip_sha256=hashes, metrics=metrics,
                foot_xy_error_cm=describe(paths),foot_orientation_deg=describe(orientation),
                leg_angle_change_rad=describe(angle_changes),added_leg_speed_rad_s=describe(velocity_changes),
                unchanged_root_xy_orientation_and_upper_joints=True, records=records)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps({k:v for k,v in report.items() if k not in ("records","clip_sha256")}), flush=True)


if __name__ == "__main__":
  main()
