"""Offline, lower-body retargeting of a flat-ground G1 reference to a static height field."""

import mujoco
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


class TerrainLegReference:
  """Keep the original ankle XY and orientation while restoring the nominal sole clearance."""

  def __init__(self, model, height_at):
    self.model = model
    self.data = mujoco.MjData(model)
    self.height_at = height_at
    sides = ("left", "right")
    self.ankles = [model.body(f"robot/{side}_ankle_roll_link").id for side in sides]
    self.geom = np.array([
      [g for g in range(model.ngeom) if model.geom_bodyid[g] == ankle
       and (model.geom_contype[g] or model.geom_conaffinity[g])]
      for ankle in self.ankles
    ])
    if not self.geom.size or not (model.geom_type[self.geom] == mujoco.mjtGeom.mjGEOM_CAPSULE).all():
      raise ValueError("Expected G1 ankle collision capsules on both feet")
    joint_names = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    joint_ids = np.array([
      [model.joint(f"robot/{side}_{joint}_joint").id for joint in joint_names]
      for side in sides
    ])
    self.qaddr = model.jnt_qposadr[joint_ids]
    self.daddr = model.jnt_dofadr[joint_ids]
    self.limits = model.jnt_range[joint_ids]
    self.jpos = np.empty((3, model.nv))
    self.jrot = np.empty_like(self.jpos)

  def sole_points(self):
    """Lowest sphere points along each capsule, in world coordinates."""
    d, m, ids = self.data, self.model, self.geom
    axis = d.geom_xmat[ids].reshape(*ids.shape, 3, 3)[..., :, 2]
    half = m.geom_size[ids, 1]
    points = d.geom_xpos[ids, None, :] + (axis * half[..., None])[..., None, :] * np.linspace(-1, 1, 5)[:, None]
    points[..., 2] -= m.geom_size[ids, 0, None]
    return points

  def lifts(self, poses):
    points = []
    for pose in poses:
      self.data.qpos[:] = pose
      mujoco.mj_kinematics(self.model, self.data)
      points.append(self.sole_points())
    points = np.stack(points)
    surface = self.height_at(torch.tensor(points[..., :2], dtype=torch.float32)).cpu().numpy()
    z = points[..., 2]
    return z.min(axis=(-1, -2)) - (z - surface).min(axis=(-1, -2))

  def adapt(self, pose, lift):
    m, d = self.model, self.data
    if not np.any(lift):
      return pose.copy()
    d.qpos[:] = pose
    mujoco.mj_kinematics(m, d)
    target_pos = d.xpos[self.ankles].copy()
    target_pos[:, 2] += lift
    target_rot = d.xmat[self.ankles].reshape(2, 3, 3).copy()
    # The lower of the two lifts keeps either leg from needing extra extension.
    d.qpos[2] += min(lift)
    for _ in range(14):
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      deltas = []
      for side, ankle in enumerate(self.ankles):
        dp = target_pos[side] - d.xpos[ankle]
        rot = target_rot[side] @ d.xmat[ankle].reshape(3, 3).T
        dr = Rotation.from_matrix(rot).as_rotvec()
        if np.linalg.norm(dp) < 5e-5 and np.linalg.norm(dr) < 0.0005:
          deltas.append(None)
          continue
        mujoco.mj_jacBody(m, d, self.jpos, self.jrot, ankle)
        jac = np.vstack((self.jpos, 0.16 * self.jrot))[:, self.daddr[side]]
        diff = np.concatenate((dp, 0.16 * dr))
        step = np.linalg.solve(jac.T @ jac + 0.0001 * np.eye(6), jac.T @ diff)
        deltas.append(np.clip(step, -0.18, 0.18))
      if all(delta is None for delta in deltas):
        break
      for side, delta in enumerate(deltas):
        if delta is not None:
          addr = self.qaddr[side]
          # Keep a pre-existing reference violation from introducing a discontinuity.
          lo = np.minimum(self.limits[side, :, 0], pose[addr])
          hi = np.maximum(self.limits[side, :, 1], pose[addr])
          d.qpos[addr] = np.clip(d.qpos[addr] + delta, lo, hi)
    return d.qpos.copy()

  def clip(self, poses, smoothing_frames=2.0):
    """Filter only the terrain correction within this clip, preserving recorded motion timing."""
    lifts = self.lifts(poses)
    adjusted = np.stack([self.adapt(q, dz) for q, dz in zip(poses, lifts, strict=True)])
    if smoothing_frames > 0:
      addr = np.r_[2, self.qaddr.ravel()]
      delta = adjusted[:, addr] - poses[:, addr]
      adjusted[:, addr] = poses[:, addr] + gaussian_filter1d(delta, smoothing_frames, axis=0, mode="nearest")
      joint = self.qaddr.ravel()
      lo = np.minimum(self.limits[..., 0].ravel(), poses[:, joint])
      hi = np.maximum(self.limits[..., 1].ravel(), poses[:, joint])
      adjusted[:, joint] = np.clip(adjusted[:, joint], lo, hi)
    return adjusted
