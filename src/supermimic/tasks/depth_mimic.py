"""Perceptive G1 walking from a moving pinhole range camera and a raw motion."""

import math
from dataclasses import dataclass

import torch
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat


@dataclass
class WalkingCameraPattern(PinholeCameraPatternCfg):
  """Pelvis-mounted camera oriented forward and down, returning metric ray ranges."""

  width: int = 48
  height: int = 32
  fovy: float = 78.0
  pitch_down: float = 46.0

  def generate_rays(self, mj_model, device):
    origins, camera = super().generate_rays(mj_model, device)
    angle = math.radians(self.pitch_down)
    s, c = math.sin(angle), math.cos(angle)
    right, up, back = camera.unbind(-1)
    directions = torch.stack([s * up - c * back, -right, c * up + s * back], -1)
    return origins + origins.new_tensor([0.12, 0, 0.15]), directions


def walking_camera():
  return RayCastSensorCfg(
    name="walk_depth",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    pattern=WalkingCameraPattern(),
    ray_alignment="base",
    include_geom_groups=(0,),
    max_distance=2.5,
    debug_vis=False,
  )


class DepthMemory:
  """Fuse range pixels in a short-lived, bounded XY grid with a cell identity tag."""

  def __init__(self, count, device, step_dt, pattern):
    self.origins, self.directions = pattern.generate_rays(None, device)
    self.resolution = 0.06
    self.width = 96  # larger than twice the camera range
    self.max_age = round(0.8 / step_dt)
    self.age = torch.full(
      (count, self.width**2), -self.max_age - 1, device=device, dtype=torch.int32
    )
    self.z = torch.zeros_like(self.age, dtype=torch.float)
    self.x = torch.zeros_like(self.age)
    self.y = torch.zeros_like(self.age)
    self.tick = 0

  def reset(self, done):
    self.age.masked_fill_(done.reshape(-1, 1), -self.max_age - 1)

  def _address(self, xy):
    return xy[..., 0].remainder(self.width) * self.width + xy[..., 1].remainder(
      self.width
    )

  def update(self, ranges, pos, quat):
    self.tick += 1
    self.pos = pos.clone()
    self.yaw = yaw_quat(quat)
    valid = (ranges > 0) & torch.isfinite(ranges) & (ranges <= 2.5)
    rays = self.origins + self.directions * torch.where(valid, ranges, 0.0)[..., None]
    cloud = pos[:, None, :] + quat_apply(
      quat[:, None, :].expand(-1, rays.shape[1], -1), rays
    )
    cells = (cloud[..., :2] / self.resolution).round().long()
    address = self._address(cells)
    # Pick the highest visible point in each occupied cell; misses never erase hits.
    z = torch.full_like(self.z, -torch.inf)
    z.scatter_reduce_(
      1, address, cloud[..., 2].masked_fill(~valid, -torch.inf), reduce="amax"
    )
    known = z.isfinite()
    self.z = torch.where(known, z, self.z)
    self.age = torch.where(known, self.tick, self.age)
    # In one camera frame no two valid points a grid width apart can collide.
    for axis, tag in enumerate((self.x, self.y)):
      incoming = torch.full_like(tag, torch.iinfo(tag.dtype).min)
      incoming.scatter_reduce_(
        1,
        address,
        cells[..., axis].int().masked_fill(~valid, torch.iinfo(tag.dtype).min),
        reduce="amax",
      )
      tag.copy_(torch.where(known, incoming, tag))

  def at(self, local_xy):
    shape = local_xy.shape[:-1]
    xy = local_xy.reshape(len(self.z), -1, 2)
    local = torch.nn.functional.pad(xy, (0, 1))
    world = self.pos[:, None, :] + quat_apply(
      self.yaw[:, None, :].expand(-1, local.shape[1], -1), local
    )
    frac = world[..., :2] / self.resolution
    base = frac.floor().long()
    corners = base[:, :, None, :] + base.new_tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    address = self._address(corners)

    def gather(board):
      return board.gather(1, address.flatten(1)).view_as(address)

    known = (
      (self.tick - gather(self.age) <= self.max_age)
      & (gather(self.x) == corners[..., 0])
      & (gather(self.y) == corners[..., 1])
    )
    weights = (1 - (corners - frac[:, :, None, :]).abs()).prod(-1) * known
    coverage = weights.sum(-1)
    valid = coverage > 0.25
    z = (gather(self.z) * weights).sum(-1) / coverage.clamp_min(1.0e-6)
    return torch.where(valid, z, 0.0).reshape(shape), valid.reshape(shape)


class DepthBalanceCorrection:
  """Gate a path-tracking ankle correction by the observed relief of the upcoming terrain."""

  def __init__(self, names, scale):
    self.pitch = [names.index(s + "_ankle_pitch_joint") for s in ("left", "right")]
    self.roll = [names.index(s + "_ankle_roll_joint") for s in ("left", "right")]
    self.scale = scale.flatten()
    self.activity = 0.0

  def terrain_gate(self, heights, valid):
    # Local change in ground elevation, insensitive to an arbitrary constant offset.
    high = heights.masked_fill(~valid, -torch.inf).amax(-1)
    low = heights.masked_fill(~valid, torch.inf).amin(-1)
    relief = torch.where(valid.sum(-1) >= 3, high - low, 0.0)
    gate = ((relief - 0.004) / 0.025).clamp(0, 1)
    self.activity = (
      torch.maximum(gate, self.activity * 0.94)
      if isinstance(self.activity, torch.Tensor)
      else gate
    )
    return self.activity

  def apply(self, action, gate, root, command, gain):
    yaw = yaw_quat(root[:, 3:7])
    err = quat_apply_inverse(yaw, command.anchor_pos_w - command.robot_anchor_pos_w)
    vel = quat_apply_inverse(
      yaw, command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w
    )
    tilt = (gain * (err[:, :2] + 0.18 * vel[:, :2])).clamp(-0.15, 0.15) * gate[:, None]
    result = action.clone()
    result[:, self.pitch] -= tilt[:, :1] / self.scale[self.pitch]
    result[:, self.roll] += tilt[:, 1:2] / self.scale[self.roll]
    return result


class DepthPredictiveCorrection:
  """Identify local leg-action responses with the frozen shared dynamics model."""

  def __init__(self, cmd, wm, horizon=4):
    self.wm = wm
    self.horizon = horizon
    names = cmd.motion.fk.joint_names
    sides = ("left", "right")
    joints = ("ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "hip_yaw")
    self.joint_ids = [
      [names.index(f"{s}_{joint}_joint") for s in sides] for joint in joints
    ]
    self.feet = [cmd.cfg.body_names.index(s + "_ankle_roll_link") for s in sides]
    self.probe = 0.055  # radians
    self.regularizer = 0.8
    self.limit = 0.13  # radians per coordinate
    self.previous = None

  def reset(self, done):
    if self.previous is not None:
      self.previous.masked_fill_(done.reshape(-1, 1), 0.0)

  def correct(self, action, terrain, gate, state, root, command, gain):
    from mjlab.utils.lab_api.math import wrap_to_pi

    from supermimic.utils.gencoord_helpers import decompose_rep_state

    wm = self.wm
    feet = command.body_pos_w[:, self.feet, 2]
    stance = torch.sigmoid((feet.flip(-1) - feet) / 0.032)
    weights = [
      0.35 + 0.65 * stance,
      0.35 + 0.65 * stance,
      1 - 0.7 * stance,
      1 - 0.7 * stance,
      0.4 + 0.6 * stance,
    ]
    b, j = action.shape
    dims = len(weights)
    basis = action.new_zeros(b, dims, j)
    for i, (cols, weight) in enumerate(zip(self.joint_ids, weights, strict=True)):
      basis[:, i, cols] = weight / wm.dynamics.action_scale.flatten()[cols]
    offsets = torch.cat(
      [torch.zeros_like(basis[:, :1]), self.probe * basis, -self.probe * basis], 1
    )
    count = offsets.shape[1]
    actions = (action[:, None, :] + offsets).reshape(-1, j)

    def expand(x):
      return x[:, None, :].expand(-1, count, -1).reshape(b * count, -1)

    aux, rep = expand(root), expand(state)
    ctx = aux.new_zeros(b * count, wm.sim_context.mean.numel())
    # Short local counterfactuals all see the same currently measured patch and
    # nominal context; no access to the map or per-environment randomization.
    scene = expand(terrain)
    for _ in range(self.horizon):
      delta = wm.dynamics.predict_delta(rep, actions, aux, ctx, sensed_terrain=scene)
      aux, rep = wm.representation.apply_prediction(
        aux, rep, delta, wm.representation.dt
      )
    _, gravity, lin, ang, _, _ = decompose_rep_state(rep)
    v = quat_apply(aux[:, 3:7], lin)
    w = quat_apply(aux[:, 3:7], ang)
    # Predict velocity, lean and yaw rate in a common frame. Position feedback
    # is added to the requested velocity to compensate accumulated path drift.
    prediction = torch.cat(
      [v[:, :2], 0.35 * gravity[:, :2], 0.22 * w[:, 2:3], 0.12 * w[:, :2]], -1
    ).reshape(b, count, -1)
    motion = command.motion
    t = command.time_steps
    end = motion.motion_ends[torch.searchsorted(motion.motion_ends, t, right=True)] - 1
    t = (t + self.horizon).minimum(end)
    target_v = motion.root_lin_vel_w[t, :2] + 2.0 * (
      motion.root_pos_w[t, :2] - aux.view(b, count, -1)[:, 0, :2]
    ).clamp(-0.5, 0.5)
    downward = torch.zeros_like(root[:, :3])
    downward[:, 2] = -1
    target_g = quat_apply_inverse(motion.root_quat_w[t], downward)
    ref_yaw = yaw_quat(motion.root_quat_w[t])
    pred_yaw = yaw_quat(aux.reshape(b, count, -1)[:, 0, 3:7].contiguous())
    yaw_error = wrap_to_pi(
      2
      * (
        torch.atan2(ref_yaw[:, 3], ref_yaw[:, 0])
        - torch.atan2(pred_yaw[:, 3], pred_yaw[:, 0])
      )
    )
    target_w = motion.root_ang_vel_w[t]
    target = torch.cat(
      [
        target_v,
        0.35 * target_g[:, :2],
        0.22 * (target_w[:, 2:3] + 2 * yaw_error[:, None]),
        0.12 * target_w[:, :2],
      ],
      -1,
    )
    jac = (prediction[:, 1 : 1 + dims] - prediction[:, 1 + dims :]) / (2 * self.probe)
    gram = jac @ jac.transpose(1, 2)
    gram.diagonal(dim1=1, dim2=2).add_(self.regularizer)
    desired = (target - prediction[:, 0]).clamp(-1.0, 1.0)
    control, info = torch.linalg.solve_ex(gram, jac @ desired[:, :, None])
    control = control.squeeze(-1)
    control = torch.where((info == 0)[:, None] & control.isfinite(), control, 0.0)
    control = (gain * control).clamp(-self.limit, self.limit) * gate[:, None]
    if self.previous is not None:
      control = 0.55 * control + 0.45 * self.previous
    self.previous = control
    return action + (control[:, None, :] @ basis).squeeze(1)


def steer_motion_window(goal, root, command, gate, steering_gain):
  """Recover direction of travel through the motion goal when depth reveals a slope.

  Align the commanded path tangent with the direction needed to rejoin the path.
  Facing and travel can differ, so apply only their angular difference to the
  original body headings (including for back- and side-stepping clips).
  """
  from mjlab.utils.lab_api.math import quat_mul

  from supermimic.representations import assemble_anchor_window, motion_anchor_window

  positions, orientations = motion_anchor_window(command.motion, command.time_steps)
  stride = positions[:, 4, :2] - positions[:, 0, :2]
  toward = (
    positions[:, 4, :2]
    + command.anchor_pos_w[:, :2]
    - positions[:, 0, :2]
    - root[:, :2]
  )
  cross = stride[:, 0] * toward[:, 1] - stride[:, 1] * toward[:, 0]
  dot = (stride * toward).sum(-1)
  angle = torch.atan2(cross, dot).clamp(-0.65, 0.65) * gate * steering_gain
  angle = angle * (stride.norm(dim=-1) / 0.25).clamp(0.0, 1.0)
  half = angle * 0.5
  zero = torch.zeros_like(half)
  zrot = torch.stack([half.cos(), zero, zero, half.sin()], -1)
  rotation = quat_mul(zrot[:, None, :].expand_as(orientations), orientations)
  _, ori = assemble_anchor_window(root[:, :3], root[:, 3:7], positions, rotation)
  return torch.cat(
    [
      goal[:, : -ori.shape[1]],
      torch.where(angle[:, None] != 0.0, ori, goal[:, -ori.shape[1] :]),
    ],
    -1,
  )


class ContactDepthAdapter:
  """Preview the orientation and elevation of each upcoming reference foothold."""

  def __init__(self, command, scale):
    motion = command.motion
    self.feet = [
      command.cfg.body_names.index(side + "_ankle_roll_link")
      for side in ("left", "right")
    ]
    self.cols = [
      [
        motion.fk.joint_names.index(side + "_" + joint + "_joint")
        for side in ("left", "right")
      ]
      for joint in ("ankle_pitch", "ankle_roll", "hip_pitch", "knee")
    ]
    self.scale = scale.flatten()
    pos, rot, *_ = motion.body_state(
      torch.arange(len(motion.joint_pos), device=self.scale.device)
    )
    self.xyz = pos[:, self.feet]
    self.foot_yaw = yaw_quat(rot[:, self.feet].contiguous())
    self.samples = pos.new_tensor(
      [
        [0.0, 0.0, 0.0],
        [0.10, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [0.0, 0.045, 0.0],
        [0.0, -0.045, 0.0],
      ]
    )

  def footprint(self, memory, command):
    motion = command.motion
    now = command.time_steps
    end = (
      motion.motion_ends[torch.searchsorted(motion.motion_ends, now, right=True)] - 1
    )
    ahead = (now + 4).minimum(end)
    ori = self.foot_yaw[ahead, :, None, :].expand(-1, -1, len(self.samples), -1)
    sole = self.xyz[ahead, :, None, :] + quat_apply(
      ori, self.samples.expand(*ori.shape[:-1], 3)
    )
    yaw = yaw_quat(motion.root_quat_w[now])
    local = quat_apply_inverse(
      yaw[:, None, None, :].expand_as(ori), sole - motion.root_pos_w[now, None, None, :]
    )
    z, known = memory.at(local[..., :2])
    mid, toe, heel, left, right = z.unbind(-1)
    fore = torch.where(known[..., 1] & known[..., 2], (toe - heel) / 0.15, 0.0)
    cross = torch.where(known[..., 3] & known[..., 4], (left - right) / 0.09, 0.0)
    return fore, cross, mid, known, self.xyz[ahead, :, 2]

  @staticmethod
  def stride_room(footprint, gain):
    """Attenuate stride per leg as its requested ankle correction approaches 0.12 rad."""
    fore, cross, _, known, _ = footprint
    required = abs(gain) * torch.maximum(fore.abs(), cross.abs())
    return (1.0 - (required / 0.12).square()).clamp(0.0, 1.0) * known.all(dim=-1)

  def apply(self, actions, memory, command, gain, *, footprint=None):
    if footprint is None:
      footprint = self.footprint(memory, command)
    fore, cross, mid, known, foot_z = footprint
    # Very steep banks correspond to edges: avoid fitting an ankle plane across a step.
    fore = torch.where(fore.abs() < 0.55, fore, 0.0)
    cross = torch.where(cross.abs() < 0.55, cross, 0.0)
    support = torch.sigmoid((foot_z.flip(-1) - foot_z) / 0.03)
    pitch, roll, hip, knee = self.cols
    corrections = actions.clone()
    corrections[:, pitch] += (-gain * support * fore).clamp(-0.12, 0.12) / self.scale[
      pitch
    ]
    corrections[:, roll] += (gain * support * cross).clamp(-0.12, 0.12) / self.scale[
      roll
    ]
    # Raise the leading swing leg only for the remaining positive inter-foot step.
    rise = torch.where(
      known[..., 0].all(-1, keepdim=True), (mid - mid.flip(-1)).clamp(0.0, 0.09), 0.0
    )
    bend = (gain * 3.0 * (1 - support) * rise).clamp(-0.18, 0.18)
    corrections[:, knee] += bend / self.scale[knee]
    corrections[:, hip] -= 0.5 * bend / self.scale[hip]
    corrections[:, pitch] -= 0.5 * bend / self.scale[pitch]
    return corrections


class ShuffledDepthControl:
  """Same reference, state and policy, paired with another window's local depth map."""

  def __init__(self, memory):
    self.memory = memory

  def at(self, xy):
    z, valid = self.memory.at(xy.roll(1, dims=0))
    return z.roll(-1, dims=0), valid.roll(-1, dims=0)


class DepthTrackingController:
  """Runtime interface shared by the camera policy and its component ablations."""

  def __init__(
    self,
    command,
    wm,
    memory,
    mode="depth-contact-stride",
    *,
    foot_gain=1.5,
    steering_gain=0.45,
    balance_gain=0.5,
    prediction_steps=4,
    stride_gain=None,
  ):
    self.command = command
    self.memory = memory
    self.view = ShuffledDepthControl(memory) if mode.endswith("-shuffled") else memory
    self.balance = DepthBalanceCorrection(
      command.motion.fk.joint_names, wm.dynamics.action_scale
    )
    self.foot = (
      ContactDepthAdapter(command, wm.dynamics.action_scale)
      if mode.startswith("depth-contact")
      else None
    )
    self.predictor = (
      DepthPredictiveCorrection(command, wm, prediction_steps)
      if mode.startswith("depth-predict")
      else None
    )
    self.steer = (
      mode.startswith(("depth-contact", "depth-steer"))
      or mode == "depth-proprio-stride"
    )
    self.use_balance = self.steer or mode == "depth-balance"
    self.unconditional = mode in (
      "depth-steer-blind",
      "depth-steer-always",
      "depth-predict-no-depth",
      "depth-proprio-stride",
    )
    self.blind = mode in (
      "depth-blind",
      "depth-steer-blind",
      "depth-predict-no-depth",
      "depth-proprio-stride",
    )
    self.use_ankle_budget = mode.startswith("depth-contact-budget")
    self.footprint = None
    self.sustained_progress = mode.startswith("depth-contact-recover")
    self.filtered_progress = None
    self.foot_gain = foot_gain
    self.stride_gain = (
      (
        1.2
        if ("stride" in mode or self.sustained_progress or self.use_ankle_budget)
        else 0.0
      )
      if stride_gain is None
      else stride_gain
    )
    self.steering_gain = steering_gain
    self.balance_gain = balance_gain

  def observe(self, ranges, position, quaternion, offsets):
    self.memory.update(ranges, position, quaternion)
    heights, valid = self.view.at(offsets)
    self.gate = self.balance.terrain_gate(heights, valid)
    if self.unconditional:
      self.gate = torch.ones_like(self.gate)
    if self.use_ankle_budget:
      self.footprint = self.foot.footprint(self.view, self.command)
    terrain = torch.zeros_like(heights) if self.blind else 10 * heights
    return terrain, heights, valid

  def goal(self, raw_goal, root):
    if self.stride_gain:
      gate = self.gate
      if self.sustained_progress:
        speed = self.command.anchor_lin_vel_w[:, :2]
        direction = speed / speed.norm(dim=-1, keepdim=True).clamp_min(0.2)
        progress = ((self.command.anchor_pos_w[:, :2] - root[:, :2]) * direction).sum(
          -1
        )
        if self.filtered_progress is None:
          self.filtered_progress = torch.zeros_like(progress)
        self.filtered_progress = torch.lerp(self.filtered_progress, progress, 0.12)
        # Recover sustained along-track drift between half and one path tolerance.
        gate = gate * ((self.filtered_progress.abs() - 0.20) / 0.20).clamp(0.0, 1.0)
      room = (
        self.foot.stride_room(self.footprint, self.foot_gain)
        if self.use_ankle_budget
        else None
      )
      raw_goal = regulate_stride_window(
        raw_goal, root, self.command, gate, self.stride_gain, leg_room=room
      )
    if self.steer:
      return steer_motion_window(
        raw_goal, root, self.command, self.gate, self.steering_gain
      )
    return raw_goal

  def action(self, raw_action, root, state, terrain):
    action = raw_action
    if self.use_balance:
      action = self.balance.apply(
        action, self.gate, root, self.command, self.balance_gain
      )
    if self.foot is not None:
      action = self.foot.apply(
        action, self.view, self.command, self.foot_gain, footprint=self.footprint
      )
    if self.predictor is not None:
      action = self.predictor.correct(
        action, terrain, self.gate, state, root, self.command, self.balance_gain
      )
    return action

  def reset(self, done):
    self.memory.reset(done)
    if self.filtered_progress is not None:
      self.filtered_progress.masked_fill_(done.reshape(-1), 0.0)
    if self.predictor is not None:
      self.predictor.reset(done)
    if isinstance(self.balance.activity, torch.Tensor):
      self.balance.activity.masked_fill_(done.reshape(-1), 0.0)


def regulate_stride_window(goal, root, command, gate, gain, *, leg_room=None):
  """Scale the hip excursion of the raw motion when along-path progress drifts."""
  from supermimic.representations import goal_window_indices

  motion = command.motion
  idx = goal_window_indices(motion, command.time_steps)
  speed = command.anchor_lin_vel_w[:, :2]
  direction = speed / speed.norm(dim=-1, keepdim=True).clamp_min(0.2)
  err = command.anchor_pos_w[:, :2] - root[:, :2]
  velocity_error = speed - command.robot_anchor_lin_vel_w[:, :2]
  signed = ((err + 0.2 * velocity_error) * direction).sum(-1).clamp(-0.55, 0.75) * gate
  factor = (gain * signed).clamp(-0.4, 0.65)
  result = goal.clone()
  nj = motion.joint_pos.shape[1]
  frames = result[:, : idx.shape[1] * 2 * nj].unflatten(-1, (idx.shape[1], 2, nj))
  cols = [
    motion.fk.joint_names.index(s + "_" + j + "_joint")
    for s in ("left", "right")
    for j in ("hip_pitch", "hip_roll")
  ]
  q = frames[:, :, 0, cols]
  # Keep the mean posture and all upper-body channels of the demonstrated motion.
  hip_factor = factor[:, None, None]
  if leg_room is not None:
    hip_factor = hip_factor * leg_room.repeat_interleave(2, dim=-1)[:, None, :]
  frames[:, :, 0, cols] = q + hip_factor * (q - q.mean(1, keepdim=True))
  frames[:, :, 1, cols] *= 1 + hip_factor
  return result
