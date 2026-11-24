"""Box-relative palm and swing-foot corrections to a G1 tracking policy."""

import re

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, get_spec
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from supermimic.representations import (
  GOAL_FRAME_STRIDE,
  NUM_GOAL_FRAMES,
  _assemble_goal,
)


def stopping_reference(motion, phase, last_frame, root, box, goal):
  """Slow the complete reference window when the box approaches its requested stop."""
  forward_speed = box[:, 7].clamp_min(0.0)
  free_travel = goal[:, 0] - box[:, 0] - 0.045 - 0.23 * forward_speed
  rate = (free_travel / 0.19).clamp(0.0, 1.0)
  offsets = torch.arange(NUM_GOAL_FRAMES, device=box.device) * GOAL_FRAME_STRIDE
  # Near the goal, decelerate the future joint targets as well as the current phase.
  horizon = (free_travel.clamp_min(0.0) / 0.009).unsqueeze(-1)
  ticks = offsets * rate[:, None]
  idx = torch.minimum(
    (phase[:, None] + torch.minimum(ticks, horizon)).long(), last_frame[:, None]
  )
  derivative = rate[:, None] * (ticks < horizon) * (idx < last_frame[:, None])
  joints = torch.cat(
    [motion.joint_pos[idx], motion.joint_vel[idx] * derivative[..., None]], -1
  )
  wm_goal, _ = _assemble_goal(
    joints.flatten(1),
    root[:, :3],
    root[:, 3:7],
    motion.root_pos_w[idx],
    motion.root_quat_w[idx],
  )
  return wm_goal, rate


class PushGuidance:
  # Hand gain, pressure offset (m), foot gain, waist lean (rad), cruise rate, brake strength.
  default_gains = (0.65, 0.0, 0.0, 0.0, 1.0, 0.68)

  def __init__(self, paths, action_scale, device):
    model = get_spec().compile()
    data = mujoco.MjData(model)
    jac = np.empty((3, model.nv))
    arms_inv, legs_inv, torque_maps = [], [], []
    for path in paths:
      with np.load(path) as f:
        names = f["joint_names"].tolist()
        poses = np.zeros((len(f["root_pos_w"]), model.nq))
        poses[:, :3], poses[:, 3:7] = f["root_pos_w"], f["root_quat_w"]
        poses[:, [model.joint(x).qposadr[0] for x in names]] = f["joint_pos"]
      arms = [
        [
          i
          for i, name in enumerate(names)
          if name.startswith(side + "_")
          and any(s in name for s in ("elbow", "shoulder", "wrist"))
        ]
        for side in ("left", "right")
      ]
      legs = [
        [
          i
          for i, name in enumerate(names)
          if name.startswith(side + "_")
          and any(s in name for s in ("hip", "knee", "ankle"))
        ]
        for side in ("left", "right")
      ]
      cols = [
        [model.joint(names[i]).dofadr[0] for i in chain] for chain in (*arms, *legs)
      ]
      gains = {
        name: act.stiffness
        for act in G1_ARTICULATION.actuators
        for name in names
        if any(re.fullmatch(pat, name) for pat in act.target_names_expr)
      }
      kp = np.array([[gains[names[i]] for i in chain] for chain in arms])
      toes = [model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
      sites = [model.site(s + "_palm").id for s in ("left", "right")]
      for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        maps = []
        force_maps = []
        for k, v in enumerate(cols):
          if k < 2:
            mujoco.mj_jacSite(model, data, jac, None, sites[k])
          else:
            body = toes[k - 2]
            point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.array(
              [0.132, 0.0, -0.025]
            )
            mujoco.mj_jac(model, data, jac, None, point, body)
          j = jac[:, v]
          if k < 2:
            force_maps.append(j.T / kp[k, :, None])
          maps.append(np.linalg.solve(j @ j.T + 0.003 * np.eye(3), j).T)
        arms_inv.append(maps[:2])
        legs_inv.append(maps[2:])
        torque_maps.append(force_maps)
    self.arm_inverse = torch.as_tensor(
      np.asarray(arms_inv), dtype=torch.float32, device=device
    )
    self.leg_inverse = torch.as_tensor(
      np.asarray(legs_inv), dtype=torch.float32, device=device
    )
    self.arms = torch.as_tensor(arms, device=device)
    self.legs = torch.as_tensor(legs, device=device)
    self.waist = names.index("waist_pitch_joint")
    self.force_maps = torch.as_tensor(
      np.asarray(torque_maps), dtype=torch.float32, device=device
    )
    self.scale = action_scale.reshape(29)

  def push_effort(self, box, palms, goal, frame, load, newtons):
    """Convert desired palm force to position-servo targets through J^T/Kp."""
    remaining = goal - box[:, :2]
    direction = remaining / remaining.norm(dim=-1, keepdim=True).clamp_min(0.05)
    speed = (box[:, 7:9] * direction).sum(-1)
    speed_target = (remaining[:, 0] * 1.6).clamp(0.0, 0.42)
    force = (load + 60.0 * (speed_target - speed)).clamp(0.0, newtons)
    force *= ((remaining[:, 0] - 0.04) / 0.10).clamp(0.0, 1.0)
    q = box[:, None, 3:7].expand(-1, 2, -1)
    local = quat_apply_inverse(q, palms - box[:, None, :3])
    near = (1.0 - (local[:, :, 0] + 0.34).abs() / 0.19).clamp(0.0, 1.0)
    near *= (local[:, :, 1].abs() < 0.3) & (local[:, :, 2].abs() < 0.45)
    planar = torch.cat([direction, torch.zeros_like(direction[:, :1])], -1)
    wrenches = (0.5 * force[:, None] * near)[:, :, None] * planar[:, None]
    dq = torch.zeros((len(box), 29), device=box.device, dtype=box.dtype)
    dq[:, self.arms] = (
      (self.force_maps[frame] @ wrenches.unsqueeze(-1)).squeeze(-1).clamp(-0.3, 0.3)
    )
    return dq / self.scale

  def command(self, box, palms, toes, toe_velocity, goal, frame, gains):
    remaining = goal - box[:, :2]
    vx = box[:, 7]
    braking = torch.sigmoid(
      (0.105 + vx.clamp_min(0.0) * 0.14 - remaining[:, 0]) / 0.035
    )
    speed_error = (1.4 * remaining[:, 0]).clamp(0.0, 0.5) - vx
    q = box[:, None, 3:7].expand(-1, 2, -1)
    local = box.new_tensor([[-0.385, 0.15, 0.34], [-0.385, -0.15, 0.34]]).expand(
      len(box), -1, -1
    )
    target = box[:, None, :3] + quat_apply(q, local)
    target_x = (
      target[:, :, 0]
      + (0.07 * speed_error + gains[:, 1] * (1 - braking) - 0.13 * braking)[:, None]
    )
    target_y = (
      target[:, :, 1]
      + (0.45 * remaining[:, 1] - 0.13 * box[:, 8]).clamp(-0.08, 0.08)[:, None]
    )
    delta = (torch.stack([target_x, target_y, target[:, :, 2]], -1) - palms).clamp(
      -0.085, 0.085
    )
    arm_dq = (
      (self.arm_inverse[frame] @ delta.unsqueeze(-1)).squeeze(-1).clamp(-0.22, 0.22)
    )
    lead_toe = toes + 0.09 * toe_velocity
    tlocal = quat_apply_inverse(q, lead_toe - box[:, None, :3])
    close = (tlocal[:, :, 0] + 0.34 + 0.10).clamp(0.0, 0.14)
    close *= (tlocal[:, :, 1].abs() < 0.34) & (tlocal[:, :, 0] < 0)
    back = torch.zeros_like(toes)
    back[:, :, 0] = -close
    leg_dq = (
      (self.leg_inverse[frame] @ quat_apply(q, back).unsqueeze(-1))
      .squeeze(-1)
      .clamp(-0.22, 0.22)
    )
    dq = torch.zeros((len(box), 29), device=box.device, dtype=box.dtype)
    dq[:, self.arms] = gains[:, 0, None, None] * arm_dq
    dq[:, self.legs] = gains[:, 2, None, None] * leg_dq
    dq[:, self.waist] = gains[:, 3] * (1 - braking)
    rate = (gains[:, 4] - gains[:, 5] * braking).clamp(0.3, 1.25)
    return dq / self.scale, rate
