"""Adapt an unmodified G1 motion window from the current pelvis-aligned range scan."""

import torch
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_apply, yaw_quat
from torch.nn import functional as F

from supermimic.representations import goal_window_indices

# In pelvis yaw coordinates the scan covers x=[-.8,1.4], y=[-.8,.8].
SCAN_SIZE = (2.2, 1.6)
SCAN_CENTER = (0.3, 0.0)
RESOLUTION = 0.05
SCAN_WIDTH, SCAN_HEIGHT = (round(size / RESOLUTION) + 1 for size in SCAN_SIZE)


def walking_scan():
  return RayCastSensorCfg(
    name="walk_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    pattern=GridPatternCfg(size=SCAN_SIZE, offset=SCAN_CENTER, resolution=RESOLUTION),
    ray_alignment="yaw",
    include_geom_groups=(0,),
    max_distance=2.0,
    debug_vis=False,
  )


class LocalGroundScan:
  def __init__(self, data):
    valid = (data.distances >= 0) & torch.isfinite(data.hit_pos_w[..., 2])
    z = torch.where(valid, data.hit_pos_w[..., 2], 0.0)
    self.image = torch.stack([z, valid.to(z.dtype)], 1).reshape(
      -1, 2, SCAN_HEIGHT, SCAN_WIDTH
    )

  def at(self, xy):
    """Bilinearly interpolate valid hits. Unknown locations return height zero and valid=False."""
    batch = xy.shape[0]
    grid = (xy - xy.new_tensor(SCAN_CENTER)) / (xy.new_tensor(SCAN_SIZE) * 0.5)
    pair = F.grid_sample(self.image, grid.reshape(batch, 1, -1, 2), align_corners=True)
    z, weight = pair.reshape(batch, 2, *xy.shape[1:-1]).unbind(1)
    known = (weight > 0.5) & (grid.abs().amax(-1) <= 1.0)
    return torch.where(known, z / weight.clamp_min(1.0e-6), 0.0), known


class TerrainMotionWindow:
  """Terrain-independent G1 leg kinematics cached from the original reference.

  A coordinated hip/knee/ankle bend shortens each leg while approximately
  preserving foot pitch. All terrain-dependent changes are computed online.
  """

  def __init__(self, command, dt):
    motion = command.motion
    self.motion = motion
    self.dt = dt
    q = motion.joint_pos
    self.cols = torch.tensor(
      [
        [
          motion.fk.joint_names.index(side + "_" + joint + "_joint")
          for joint in ("hip_pitch", "knee", "ankle_pitch")
        ]
        for side in ("left", "right")
      ],
      device=q.device,
    )
    ankles = [
      command.cfg.body_names.index(side + "_ankle_roll_link")
      for side in ("left", "right")
    ]
    self.direction = q.new_tensor([-0.5, 1.0, -0.5])
    limits = command.robot.data.soft_joint_pos_limits[0, self.cols]
    orig = q[:, self.cols]
    travel = torch.where(
      self.direction > 0,
      (limits[..., 1] - orig) / self.direction,
      (limits[..., 0] - orig) / self.direction,
    )
    self.max_bend = travel.amin(-1).clamp(0.0, 1.15)
    samples = q.new_tensor(
      [
        [-0.055, 0.0, -0.025],
        [0.07, 0.035, -0.025],
        [0.07, -0.035, -0.025],
        [0.14, 0.0, -0.025],
      ]
    )
    self.levels = 13
    xyz, quat, *_ = motion.body_state(torch.arange(len(q), device=q.device))
    pos = xyz[:, ankles]
    ori = quat[:, ankles, None, :].expand(-1, -1, len(samples), -1)
    self.foot_xy = (
      pos[:, :, None, :] + quat_apply(ori, samples.expand(*ori.shape[:-1], 3))
    )[..., :2]
    heights = []
    for k in range(self.levels):
      pose = q.clone()
      pose[:, self.cols] += (
        self.max_bend[..., None] * (k / (self.levels - 1)) * self.direction
      )
      bent = motion.fk(
        motion.root_pos_w,
        motion.root_quat_w,
        pose,
        motion.root_lin_vel_w,
        motion.root_ang_vel_w,
        motion.joint_vel,
      )[0]
      heights.append(bent[:, ankles, 2] - pos[..., 2])
    self.heights = torch.stack(heights, -1).cummax(-1).values

  def bend(self, indices, desired):
    table = self.heights[indices]
    desired = desired.clamp_min(0.0).minimum(table[..., -1])
    upper = (table < desired[..., None]).sum(-1).clamp(1, self.levels - 1)
    a = table.gather(-1, (upper - 1).unsqueeze(-1)).squeeze(-1)
    b = table.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    fraction = (upper - 1 + (desired - a) / (b - a).clamp_min(1.0e-6)) / (
      self.levels - 1
    )
    return self.max_bend[indices] * fraction

  def actor_goal(self, scan, times, original_goal):
    """Adjust leg positions and velocities in the current actor reference window."""
    motion = self.motion
    idx = goal_window_indices(motion, times)
    clip = torch.searchsorted(motion.motion_ends, times, right=True)
    first = F.pad(motion.motion_ends, (1, 0))[clip]
    last = motion.motion_ends[clip] - 1
    neighbors = torch.stack(
      [(idx - 1).maximum(first[:, None]), idx, (idx + 1).minimum(last[:, None])], -1
    )
    # Register the commanded path at the measured pelvis. Sensor pixels and
    # nominal foot displacements are expressed in their respective yaw frames.
    xy = self.foot_xy[neighbors] - motion.root_pos_w[times, None, None, None, None, :2]
    yaw = yaw_quat(motion.root_quat_w[times])
    c = 1 - 2 * yaw[:, 3].square()
    s = 2 * yaw[:, 0] * yaw[:, 3]
    shape = (-1, 1, 1, 1, 1)
    x, y = xy.unbind(-1)
    foot = torch.stack(
      [c.view(shape) * x + s.view(shape) * y, -s.view(shape) * x + c.view(shape) * y],
      -1,
    )
    h, known = scan.at(foot)
    # Unknown footprints keep the original target. The lower support foot
    # sets the common pelvis lift; only the inter-foot difference bends a leg.
    surface = h.masked_fill(~known, -torch.inf).amax(-1)
    surface = torch.where(known.any(-1).all(-1, keepdim=True), surface, 0.0)
    pelvis = surface.amin(-1, keepdim=True)
    amount = self.bend(neighbors, surface - pelvis)
    delta = amount[..., None] * self.direction
    goal = original_goal.clone()
    width = motion.joint_pos.shape[-1]
    joints = goal[:, : idx.shape[1] * 2 * width].unflatten(-1, (idx.shape[1], 2, width))
    q, v = joints.unbind(2)
    q[:, :, self.cols] += delta[:, :, 1]
    frames = (neighbors[:, :, 2] - neighbors[:, :, 0]).clamp_min(1)
    v[:, :, self.cols] += (delta[:, :, 2] - delta[:, :, 0]) / (
      frames[..., None, None] * self.dt
    )
    return goal, amount[:, :, 1].mean(-1), known[:, :, 1].float().mean((-1, -2))
