"""Depth-conditioned joint-target correction around a fixed tracking policy."""

import hashlib

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from torch import nn

from supermimic.utils.helpers import extract_action_scale_offset


def adapter_inputs(obs, base_action, foot_terrain=None):
  """Build features in the robot frame, without any adapted reference inputs."""
  depth = obs["depth_tokens"]
  visible = depth[..., 3:4] > 0.5
  # Preserve the distinction between an unobserved cell and measured zero height.
  heights = torch.where(visible, depth[..., 2:3], 0)
  terrain = torch.cat((heights, visible), dim=-1).flatten(1)
  context = torch.cat(tuple(value.flatten(1) for value in (
    obs["goal_motion"], obs["history"], obs["future_feet"], base_action,
  )), dim=-1)
  parts = (context, terrain) if foot_terrain is None else (context, terrain, foot_terrain.flatten(1))
  return torch.cat(parts, dim=-1), visible.any(dim=1)


def sample_future_foot_terrain(patch, feet, motion, frames, aux, origins):
  """Query measured heights at nominal future feet and nearby sole points."""
  position, forward, lateral = feet.split(3, dim=-1)
  # Foot positions and axes initially share the current reference pelvis frame.
  offsets = feet.new_tensor([[0, 0], [0.08, 0], [-0.08, 0], [0, 0.04], [0, -0.04]])
  xy_forward, xy_lateral = offsets.split(1, dim=-1)
  points = (position[..., None, :] + forward[..., None, :] * xy_forward
            + lateral[..., None, :] * xy_lateral).flatten(1, 3)
  reference_yaw = yaw_quat(motion.root_quat_w[frames])[:, None].expand(-1, points.shape[1], -1)
  world = quat_apply(reference_yaw, points) + (motion.root_pos_w[frames] + origins)[:, None]
  current_yaw = yaw_quat(aux[:, 3:7].contiguous())[:, None].expand_as(reference_yaw)
  current = quat_apply_inverse(current_yaw, world - aux[:, None, :3])
  return patch.sample(aux, current[..., :2])


def tracking_gate(drift: torch.Tensor, limit_m: float) -> torch.Tensor:
  """Smoothly hand control back to the tracking policy as planar drift grows."""
  progress = (2.0 * drift / limit_m - 1.0).clamp(0.0, 1.0)
  return 1.0 - progress.square() * (3.0 - 2.0 * progress)


class OnlineTerrainAdapter(nn.Module):
  """Learn a bounded residual in PD target units (radians)."""

  def __init__(self, input_dim: int, num_actions: int, max_delta_rad: float):
    super().__init__()
    self.max_delta_rad = max_delta_rad
    self.net = nn.Sequential(
      nn.Linear(input_dim, 256),
      nn.LayerNorm(256),
      nn.SiLU(),
      nn.Linear(256, 128),
      nn.SiLU(),
      nn.Linear(128, num_actions),
    )
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, features: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    delta = self.max_delta_rad * self.net(features).tanh()
    return torch.where(visible, delta, 0.0)


class OnlineTerrainActions:
  """Apply a saved student to an ordinary raw-reference MuJoCo rollout."""

  def __init__(self, weights, actor_path, device, *, mask_depth=False, max_root_drift=None):
    bundle = torch.load(weights, map_location=device, weights_only=True)
    with actor_path.open("rb") as source:
      actor_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if bundle["actor_sha256"] != actor_hash:
      raise ValueError("Student must be paired with its original tracking policy")
    self.model = OnlineTerrainAdapter(bundle["input_dim"], 29, bundle["max_delta_rad"]).to(device)
    self.model.load_state_dict(bundle["adapter"])
    self.model.eval()
    self.mask_depth = mask_depth
    self.max_root_drift = max_root_drift
    self.foot_context = bundle.get("foot_context", False)

  def bind(self, sim, motion, runner, patch):
    self.scale, _ = extract_action_scale_offset(sim)
    if self.scale.shape[-1] != 29 or not (self.scale > 0).all():
      raise ValueError("Expected 29 joint targets with positive action scales")
    self.motion = motion.motion
    self.origins = sim.scene.env_origins
    self.patch = patch

  def act(self, actor, obs, representation, frames, aux, active, step):
    base_action = actor(obs)
    foot_terrain = (sample_future_foot_terrain(self.patch, obs["future_feet"], self.motion,
                                              frames, aux, self.origins)
                    if self.foot_context else None)
    features, visible = adapter_inputs(obs, base_action, foot_terrain)
    if self.mask_depth:
      visible = torch.zeros_like(visible)
    delta = self.model(features, visible)
    if self.max_root_drift is not None:
      reference_xy = self.motion.root_pos_w[frames, :2] + self.origins[:, :2]
      drift = (reference_xy - aux[:, :2]).norm(dim=-1, keepdim=True)
      delta = delta * tracking_gate(drift, self.max_root_drift)
    return base_action + delta / self.scale
