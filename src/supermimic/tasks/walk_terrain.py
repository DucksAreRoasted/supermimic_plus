"""A low-step walking course with matching MuJoCo and differentiable geometry."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class WalkTerrain:
  """Repeating ramps and narrow step edges along x; y is level across the course."""

  height: float = 0.05
  phase_cells: int = 0
  cell_size: float = 0.05
  half_cells: int = 320
  half_width: float = 16.0

  def profile(self) -> np.ndarray:
    """Unclipped elevation; low portions lie beneath the original ground plane."""
    if self.height < 0 or self.cell_size <= 0 or self.half_width <= 0:
      raise ValueError(
        "Expected a nonnegative course height and positive width and cell size"
      )
    phase = (np.arange(-self.half_cells, self.half_cells + 1) + self.phase_cells) % 80
    # Each 4 m cycle includes a long ramp, a level area and a one-cell step edge.
    return np.interp(
      phase,
      [0, 9, 25, 32, 48, 55, 56, 66, 67, 79],
      [
        -0.02,
        -0.02,
        self.height,
        self.height,
        -0.02,
        -0.02,
        self.height,
        self.height,
        -0.02,
        -0.02,
      ],
    ).astype(np.float32)

  def __call__(self, spec: mujoco.MjSpec) -> None:
    profile = self.profile()
    if self.height == 0:
      return
    (ground,) = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    lo, hi = float(profile.min()), float(profile.max())
    normalized = (profile - lo) / (hi - lo)
    field = spec.add_hfield(
      name="walking_course",
      size=[self.cell_size * self.half_cells, self.half_width, hi - lo, 0.02],
      ncol=len(profile),
      nrow=2,
      userdata=np.tile(normalized, 2).tolist(),
    )
    spec.worldbody.add_geom(
      name="walking_surface",
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=(0, 0, lo),
      contype=ground.contype,
      conaffinity=ground.conaffinity,
      friction=ground.friction,
      condim=ground.condim,
    )


class TerrainObservation(nn.Module):
  """Query the same collision height field at a real or imagined root pose (wxyz)."""

  num_features = 15

  def __init__(self, terrain: WalkTerrain):
    super().__init__()
    self.cell_size = terrain.cell_size
    self.half_width = terrain.half_width
    self.half_cells = terrain.half_cells
    self.register_buffer(
      "profile", torch.from_numpy(terrain.profile()), persistent=False
    )
    ahead = torch.tensor([-0.25, 0.0, 0.25, 0.50, 0.75])
    lateral = torch.tensor([-0.18, 0.0, 0.18])
    self.register_buffer(
      "offsets", torch.cartesian_prod(ahead, lateral), persistent=False
    )

  def height_at(self, xy: torch.Tensor) -> torch.Tensor:
    u = xy[..., 0] / self.cell_size + self.half_cells
    inside = (
      (u >= 0) & (u <= 2 * self.half_cells) & (xy[..., 1].abs() <= self.half_width)
    )
    left = u.floor().long().clamp(0, 2 * self.half_cells - 1)
    h = torch.lerp(
      self.profile[left], self.profile[left + 1], (u - left).clamp(0.0, 1.0)
    )
    return torch.where(inside, h.clamp_min(0.0), 0.0)

  def forward(self, root: torch.Tensor) -> torch.Tensor:
    w, x, y, z = root[..., 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
    dx, dy = self.offsets.unbind(-1)
    world = root[..., None, :2] + torch.stack([c * dx - s * dy, s * dx + c * dy], -1)
    return self.height_at(world)

  def features(self, root: torch.Tensor) -> torch.Tensor:
    """Use decimeters for network inputs; collision geometry stays in meters."""
    return self(root) * 10.0
