"""Retarget a flat G1 walk to the same low course used by simulation and imagination."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from supermimic.tasks.walk_terrain import WalkTerrain


def retarget(clip: dict, course: WalkTerrain) -> tuple[dict, dict]:
  model = get_spec().compile()
  data = mujoco.MjData(model)
  addresses = np.array(
    [model.joint(str(name)).qposadr[0] for name in clip["joint_names"]]
  )
  if set(addresses) != set(range(7, model.nq)):
    raise ValueError("Expected all G1 hinge joints exactly once")
  root = clip["root_pos_w"]
  joints = clip["joint_pos"]
  poses = np.zeros((len(root), model.nq))
  poses[:, :3], poses[:, 3:7], poses[:, addresses] = root, clip["root_quat_w"], joints
  feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
  caps = np.array(
    [
      [model.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)]
      for side in ("left", "right")
    ]
  )
  # Select each foot's kinematic ancestors (independent of XML or NPZ joint ordering).
  legs = []
  for foot in feet:
    limb = []
    body = foot
    while model.jnt_type[model.body(body).jntadr[0]] != mujoco.mjtJoint.mjJNT_FREE:
      limb.append(model.body(body).jntadr[0])
      body = model.body(body).parentid[0]
    legs.append(np.array(limb))
  profile = course.profile()
  grid = (np.arange(len(profile)) - course.half_cells) * course.cell_size

  def height(xy):
    return np.where(
      abs(xy[..., 1]) <= course.half_width,
      np.maximum(0.0, np.interp(xy[..., 0], grid, profile, left=0.0, right=0.0)),
      0.0,
    )

  positions = np.empty((len(root), 2, 3))
  orientations = np.empty((len(root), 2, 4))
  lifts = np.empty((len(root), 2))
  for i, pose in enumerate(poses):
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    positions[i], orientations[i] = data.xpos[feet], data.xquat[feet]
    half_axis = (
      data.geom_xmat[caps].reshape(2, 7, 3, 3)[..., :, 2]
      * model.geom_size[caps, 1, None]
    )
    center = data.geom_xpos[caps]
    samples = np.stack([center - half_axis, center, center + half_axis], axis=2)
    lifts[i] = height(samples[..., :2]).max(axis=(1, 2))

  # A short lookahead raises the swinging foot before the step edge.
  lifts = maximum_filter1d(lifts, size=5, axis=0, mode="nearest")
  # The lower foot limits how far the pelvis can rise while the trailing leg is extended.
  root_lift = gaussian_filter1d(lifts.min(axis=1), sigma=3, mode="nearest")
  poses[:, 2] += root_lift
  targets = positions.copy()
  targets[:, :, 2] += lifts
  errors = []
  jac_pos, jac_rot = np.empty((3, model.nv)), np.empty((3, model.nv))
  rot_error = np.empty(3)
  for i, pose in enumerate(poses):
    data.qpos[:] = pose
    for leg, foot in enumerate(feet):
      ids = legs[leg]
      q, v = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
      # Seed a bent-knee solution when a nearly straight leg must shorten.
      # An extended leg has almost no vertical Jacobian in the original pose.
      side = ("left", "right")[leg]
      pitch = [
        model.joint(f"{side}_{name}_joint").qposadr[0]
        for name in ("hip_pitch", "knee", "ankle_pitch")
      ]
      if lifts[i, leg] - root_lift[i] > 0.002 and data.qpos[pitch[1]] < 0.35:
        bend = 0.35 - data.qpos[pitch[1]]
        data.qpos[pitch] += np.array([-0.5, 1.0, -0.5]) * bend
      for _ in range(24):
        mujoco.mj_kinematics(model, data)
        delta = targets[i, leg] - data.xpos[foot]
        mujoco.mju_subQuat(rot_error, orientations[i, leg], data.xquat[foot])
        # mju_subQuat gives a body-frame vector; the MuJoCo angular Jacobian is world-frame.
        angular = data.xmat[foot].reshape(3, 3) @ rot_error
        if np.linalg.norm(delta) < 0.0003 and np.linalg.norm(angular) < 0.003:
          break
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, foot)
        jac = np.vstack([jac_pos[:, v], 0.12 * jac_rot[:, v]])
        residual = np.r_[delta, 0.12 * angular]
        update = jac.T @ np.linalg.solve(jac @ jac.T + 1.0e-5 * np.eye(6), residual)
        scale = min(1.0, 0.15 / max(1.0e-8, np.abs(update).max()))
        data.qpos[q] = np.clip(data.qpos[q] + scale * update, *model.jnt_range[ids].T)
      mujoco.mj_kinematics(model, data)
      errors.append(np.linalg.norm(targets[i, leg] - data.xpos[foot]))
    pose[:] = data.qpos
  out = {k: v.copy() for k, v in clip.items()}
  out["root_pos_w"][:, 2] = poses[:, 2]
  out["joint_pos"][:] = poses[:, addresses]
  fps = float(clip["fps"].item())
  out["joint_vel"] += np.gradient(out["joint_pos"] - joints, axis=0) * fps
  out["root_lin_vel_w"][:, 2] += np.gradient(root_lift) * fps
  return out, dict(
    frames=len(root),
    terrain_height_m=course.height,
    phase_cells=course.phase_cells,
    ankle_target_p95_m=float(np.quantile(errors, 0.95)),
    ankle_target_max_m=float(np.max(errors)),
    elevated_foot_fraction=float((lifts > 0.002).mean()),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--height", type=float, default=0.05)
  parser.add_argument("--phase", type=int, default=0)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--frames", type=int)
  args = parser.parse_args()
  if args.output.resolve() == args.motion.resolve():
    parser.error("Write the adapted reference to a separate file")
  with np.load(args.motion) as archive:
    n = len(archive["root_pos_w"])
    clip = {
      k: (
        v[args.start : None if args.frames is None else args.start + args.frames]
        if v.ndim and len(v) == n
        else v
      )
      for k, v in archive.items()
    }
  out, metrics = retarget(clip, WalkTerrain(args.height, args.phase))
  args.output.parent.mkdir(exist_ok=True, parents=True)
  np.savez_compressed(args.output, **out)
  print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
