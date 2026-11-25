"""Small vertical changes to reference ankles over continuous rough ground."""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat


class RoughFootReference:
  """Use one terrain snapshot for the current reward and the Actor's three future poses."""

  def __init__(self, feet, mode, ground):
    if mode not in ("camera", "oracle"):
      raise ValueError(f"Unknown rough reference source: {mode}")
    self.feet = feet
    self.body_indices = torch.as_tensor(feet.body_indices, dtype=torch.long, device=feet.xyz.device)
    self.ground = ground if mode == "oracle" else None
    self.offsets = torch.cat((feet.ahead.new_tensor([0, 1]), feet.ahead))

  @torch.no_grad()
  def targets(self, time, root, patch):
    feet = self.feet
    motion = feet.motion
    end = (
      motion.motion_ends[torch.searchsorted(motion.motion_ends, time, right=True)] - 1
    )
    frames = torch.minimum(time[:, None] + self.offsets, end[:, None])
    nominal = feet.xyz[frames]  # B, horizons, feet, xyz
    rotation = yaw_quat(motion.root_quat_w[time])[:, None, None]
    local = quat_apply_inverse(
      rotation.expand(*nominal.shape[:-1], 4),
      nominal - motion.root_pos_w[time, None, None],
    )
    xy = local[..., :2].flatten(1, 2)
    if self.ground is None:
      samples = patch.sample(root, xy)
      heights = samples[..., 0].reshape_as(local[..., 2]) * 0.1  # patch units are dm
      valid = samples[..., 1].reshape_as(heights).bool()
    else:
      yaw = yaw_quat(root[:, 3:7])[:, None].expand(-1, xy.shape[1], -1)
      world = (
        root[:, None, :2]
        + quat_apply(yaw, torch.nn.functional.pad(xy, (0, 1)))[..., :2]
      )
      heights = self.ground.height_at(world).reshape_as(local[..., 2])
      valid = torch.ones_like(heights, dtype=torch.bool)

    # Blend support continuously when the ankles are close in height. Compute the
    # baseline at each horizon so the velocity uses the same target as position.
    ankle_z = nominal[..., 2]
    support_mix = ((ankle_z[..., 1] - ankle_z[..., 0]) / 0.08 + 0.5).clamp(0, 1)
    support_mix = support_mix.square() * (3 - 2 * support_mix)
    weights = torch.stack((support_mix, 1 - support_mix), -1)
    baseline = (weights * heights).sum(-1, keepdim=True)
    support_visible = (valid | (weights == 0)).all(-1, keepdim=True)
    visible = valid & support_visible
    delta = (heights - baseline).clamp(-0.10, 0.10)
    # Continuous dead zone for submillimetre interpolation noise.
    delta = torch.where(visible, delta.sign() * (delta.abs() - 0.001).clamp_min(0), 0.0)
    # In swing, preserve nominal clearance over depressions. Blend continuously
    # as the feet approach equal height, allowing a lower upcoming touchdown.
    swing = ((nominal[..., 2] - nominal[..., 2].flip(-1)) / 0.04).clamp(0, 1)
    delta = delta - swing * delta.clamp_max(0)
    future = feet(time).clone()
    future[..., 2] += delta[:, 2:].transpose(1, 2)
    # A visibility edge is not vertical foot velocity.
    now = delta[:, 0]
    next_for_velocity = torch.where(visible[:, 0] & visible[:, 1], delta[:, 1], now)
    return future, now, next_for_velocity

  def body_targets(self, body, now, after, dt):
    pos, rot, vel, ang = body
    pos, vel = pos.clone(), vel.clone()
    pos[:, self.body_indices, 2] += now
    vel[:, self.body_indices, 2] += (after - now) / dt
    return pos, rot, vel, ang


def rough_body_tracking(env, command_name, std, velocity, body_names=None):
  """Native simulator tracking score for the same pre-step targets used in imagination."""
  command = env.command_manager.get_term(command_name)
  nominal = command.body_lin_vel_w if velocity else command.body_pos_relative_w
  actual = command.robot_body_lin_vel_w if velocity else command.robot_body_pos_w
  error = nominal - actual
  correction = getattr(command, "rough_ankle_targets", None)
  if correction is not None:
    indices, dz, vz = correction
    error[:, indices, 2] += vz if velocity else dz
  if body_names is not None:
    error = error[
      :, [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
    ]
  return torch.exp(-error.square().sum(-1).mean(-1) / std**2)


def configure_rough_rewards(env_cfg):
  for name, velocity in (("motion_body_pos", False), ("motion_body_lin_vel", True)):
    term = env_cfg.rewards[name]
    term.func = rough_body_tracking
    term.params = {**term.params, "velocity": velocity}
