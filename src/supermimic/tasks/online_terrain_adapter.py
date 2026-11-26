"""Depth-conditioned joint-target correction around a fixed tracking policy."""

import torch
from torch import nn


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
