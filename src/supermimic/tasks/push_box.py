"""A free rigid box for G1 palm-contact tasks."""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from supermimic.tasks.walk_terrain import WalkTerrain


@dataclass(frozen=True)
class PushBox:
  center: tuple[float, float, float] = (0.62, 0.0, 0.42)
  half_size: tuple[float, float, float] = (0.16, 0.27, 0.42)
  mass: float = 2.0
  friction: float = 0.55
  floor_friction: float | None = None

  def __call__(self, spec: mujoco.MjSpec) -> None:
    if self.mass <= 0 or min(self.half_size) <= 0 or self.friction < 0:
      raise ValueError("Expected a rigid box with positive mass and dimensions")
    box = spec.worldbody.add_body(name="push_box", pos=self.center)
    box.add_freejoint(name="push_box_free")
    box.add_geom(
      name="push_box_collision",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=self.half_size,
      mass=self.mass,
      friction=(self.friction, 0.005, 0.0001),
      condim=3,
      contype=1,
      conaffinity=1,
      rgba=(0.64, 0.38, 0.18, 1.0),
    )
    if self.floor_friction is not None:
      if self.floor_friction < 0:
        raise ValueError("Expected nonnegative floor friction")
      (ground,) = [
        geom for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_PLANE
      ]
      spec.add_pair(
        geomname1=ground.name,
        geomname2="push_box_collision",
        condim=3,
        friction=(self.floor_friction, self.floor_friction, 0.005, 0.0001, 0.0001),
      )
      for geom in spec.geoms:
        if geom.name.startswith("push_ledge_"):
          spec.add_pair(
            geomname1=geom.name,
            geomname2="push_box_collision",
            condim=3,
            friction=(self.floor_friction, self.floor_friction, 0.005, 0.0001, 0.0001),
          )


@dataclass(frozen=True)
class PushCourse(WalkTerrain):
  box: PushBox = PushBox()
  grade: float = 0.0

  def ledges(self):
    # Floor elevations add up to the requested height. Both ramps start on grid points.
    shift = self.phase_cells * self.cell_size
    return [
      (0.15 - shift, 0.50 - shift, 0.0, self.height / 2),
      (1.35 - shift, 1.70 - shift, self.height / 2, self.height),
    ]

  def profile(self) -> np.ndarray:
    x = np.arange(-self.half_cells, self.half_cells + 1) * self.cell_size
    z = np.zeros_like(x)
    for start, end, lower, upper in self.ledges():
      z += (upper - lower) * np.clip((x - start) / (end - start), 0.0, 1.0)
    return z.astype(np.float32)

  def __call__(self, spec: mujoco.MjSpec) -> None:
    if self.lateral_height or self.height < 0 or (self.grade and self.height):
      raise ValueError("A push task uses two longitudinal ledges or one tilted plane")
    (ground,) = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if self.grade:
      angle = -math.atan(self.grade)
      ground.quat = [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]
    elif self.height:
      # Large free bodies interacting with a finely tessellated hfield can receive
      # spurious impulses in Warp. Each shelf is a single convex contact surface.
      for i, (start, end, lower, upper) in enumerate(self.ledges()):
        right = self.half_cells * self.cell_size
        if not -right <= start < end < right:
          raise ValueError("Ledge is outside the configured course")
        outline = [
          (start, -0.03),
          (right, -0.03),
          (right, upper),
          (end, upper),
          (start, lower),
        ]
        verts = [
          (x, y, z) for y in (-self.half_width, self.half_width) for x, z in outline
        ]
        name = f"push_ledge_{i}"
        spec.add_mesh(name=name, uservert=np.asarray(verts).ravel().tolist())
        spec.worldbody.add_geom(
          name=name,
          type=mujoco.mjtGeom.mjGEOM_MESH,
          meshname=name,
          contype=ground.contype,
          conaffinity=ground.conaffinity,
          friction=ground.friction,
          condim=ground.condim,
        )
    self.box(spec)
