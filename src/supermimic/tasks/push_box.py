"""A free rigid box for G1 palm-contact tasks."""

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class PushBox:
  center: tuple[float, float, float] = (0.62, 0.0, 0.42)
  half_size: tuple[float, float, float] = (0.16, 0.27, 0.42)
  mass: float = 2.0
  friction: float = 0.55

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
