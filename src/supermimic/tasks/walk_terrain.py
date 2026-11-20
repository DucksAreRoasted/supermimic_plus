"""A low-step walking course with matching MuJoCo and differentiable geometry."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class WalkTerrain:
  """Repeating ramps and low step edges with optional transverse ridges."""

  height: float = 0.05
  phase_cells: int = 0
  cell_size: float = 0.05
  half_cells: int = 320
  half_width: float = 16.0
  lateral_height: float = 0.0

  def cross_profile(self) -> np.ndarray:
    if self.lateral_height < 0:
      raise ValueError("Cross-slope amplitude must be nonnegative")
    if not self.lateral_height:
      return np.zeros(2, dtype=np.float32)
    y = np.linspace(-self.half_width, self.half_width, 129)
    return (
      self.lateral_height * np.sin(2 * np.pi * y / 1.6 + self.phase_cells * 0.11)
    ).astype(np.float32)

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
    surface = self.profile()[None, :] + self.cross_profile()[:, None]
    if self.height == 0 and self.lateral_height == 0:
      return
    (ground,) = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    lo, hi = float(surface.min()), float(surface.max())
    normalized = (surface - lo) / (hi - lo)
    field = spec.add_hfield(
      name="walking_course",
      size=[self.cell_size * self.half_cells, self.half_width, hi - lo, 0.02],
      ncol=surface.shape[1],
      nrow=surface.shape[0],
      userdata=normalized.flatten().tolist(),
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
    self.cross_slopes = bool(terrain.lateral_height)
    self.grade = getattr(terrain, "grade", 0.0)
    self.register_buffer(
      "cross_profile", torch.from_numpy(terrain.cross_profile()), persistent=False
    )
    self.register_buffer(
      "profile", torch.from_numpy(terrain.profile()), persistent=False
    )
    ahead = torch.tensor([-0.25, 0.0, 0.25, 0.50, 0.75])
    lateral = torch.tensor([-0.18, 0.0, 0.18])
    self.register_buffer(
      "offsets", torch.cartesian_prod(ahead, lateral), persistent=False
    )

  def height_at(self, xy: torch.Tensor) -> torch.Tensor:
    if self.grade:
      return self.grade * xy[..., 0]
    u = xy[..., 0] / self.cell_size + self.half_cells
    inside = (
      (u >= 0) & (u <= 2 * self.half_cells) & (xy[..., 1].abs() <= self.half_width)
    )
    left = u.floor().long().clamp(0, 2 * self.half_cells - 1)
    profile = self.profile.to(dtype=xy.dtype)
    h = torch.lerp(profile[left], profile[left + 1], (u - left).clamp(0.0, 1.0))
    if self.cross_slopes:
      last = len(self.cross_profile) - 1
      v = (xy[..., 1] / self.half_width + 1.0) * (last / 2)
      bottom = v.floor().long().clamp(0, last - 1)
      cross = self.cross_profile.to(dtype=xy.dtype)
      h = h + torch.lerp(
        cross[bottom],
        cross[bottom + 1],
        (v - bottom).clamp(0.0, 1.0),
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
