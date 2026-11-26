"""Depth-conditioned joint-target correction around a fixed tracking policy."""

import hashlib

import torch
from torch import nn

from supermimic.utils.helpers import extract_action_scale_offset


def adapter_inputs(obs, base_action):
  """Build features in the robot frame, without any adapted reference inputs."""
  depth = obs["depth_tokens"]
  visible = depth[..., 3:4] > 0.5
  # Preserve the distinction between an unobserved cell and measured zero height.
  heights = torch.where(visible, depth[..., 2:3], 0)
  terrain = torch.cat((heights, visible), dim=-1).flatten(1)
  context = torch.cat(tuple(value.flatten(1) for value in (
    obs["goal_motion"], obs["history"], obs["future_feet"], base_action,
  )), dim=-1)
  return torch.cat((context, terrain), dim=-1), visible.any(dim=1)


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

  def __init__(self, weights, actor_path, device, *, mask_depth=False):
    bundle = torch.load(weights, map_location=device, weights_only=True)
    with actor_path.open("rb") as source:
      actor_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if bundle["actor_sha256"] != actor_hash:
      raise ValueError("Student must be paired with its original tracking policy")
    self.model = OnlineTerrainAdapter(bundle["input_dim"], 29, bundle["max_delta_rad"]).to(device)
    self.model.load_state_dict(bundle["adapter"])
    self.model.eval()
    self.mask_depth = mask_depth

  def bind(self, sim, motion, runner):
    self.scale, _ = extract_action_scale_offset(sim)
    if self.scale.shape[-1] != 29 or not (self.scale > 0).all():
      raise ValueError("Expected 29 joint targets with positive action scales")

  def act(self, actor, obs, representation, frames, aux, active, step):
    base_action = actor(obs)
    features, visible = adapter_inputs(obs, base_action)
    if self.mask_depth:
      visible = torch.zeros_like(visible)
    return base_action + self.model(features, visible) / self.scale
