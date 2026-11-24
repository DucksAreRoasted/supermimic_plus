"""Place the G1 motion reference pose above the local walking surface at reset."""

from dataclasses import fields

import torch
import warp as wp

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

from supermimic.tasks.walk_terrain import TerrainObservation


class TerrainResetMotionCommand(MotionCommand):
  def __init__(self, cfg, env):
    self.ground = TerrainObservation(env.cfg.scene.spec_fn).to(env.device)
    super().__init__(cfg, env)

  def _write_reference_state_to_sim(
    self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
  ):
    # The caller has already added the scene origin and any sampled pose noise.
    placed = root_pos.clone()
    placed[:, 2] += self.ground.height_at(placed[:, :2])
    super()._write_reference_state_to_sim(
      env_ids, placed, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )


class TerrainResetMotionCommandCfg(MotionCommandCfg):
  @classmethod
  def from_motion(cls, motion: MotionCommandCfg):
    return cls(**{f.name: getattr(motion, f.name) for f in fields(motion) if f.init})

  def build(self, env):
    return TerrainResetMotionCommand(self, env)

class TerrainHeightSchedule:
  """Apply the same terrain scale to collision, camera rays and reset queries."""

  def __init__(self, sim, motion, query):
    self.sim = sim
    self.camera = sim._sensor_context
    model = sim.mj_model
    field = model.hfield("walking_course").id
    geom = model.geom("walking_surface").id
    start = int(model.hfield_adr[field])
    count = int(model.hfield_nrow[field] * model.hfield_ncol[field])
    self.span = slice(start, start + count)
    self.full_height = model.hfield_data[self.span].copy()
    self.zero_height = -model.geom_pos[geom, 2] / model.hfield_size[field, 2]
    if not 0 <= self.zero_height <= 1:
      raise ValueError("Terrain elevation range must include the ground plane")
    self.gpu_height = wp.to_torch(sim.wp_model.hfield_data)[self.span]
    self.queries = (motion.ground, query)
    self.full_surfaces = tuple(item.surface_grid.clone() for item in self.queries)

  def apply(self, scale):
    if not 0 <= scale <= 1:
      raise ValueError("Terrain scale must be between zero and one")
    model = self.sim.mj_model
    model.hfield_data[self.span] = self.zero_height + scale * (self.full_height - self.zero_height)
    self.gpu_height.copy_(torch.as_tensor(model.hfield_data[self.span], device=self.gpu_height.device))
    for query, full in zip(self.queries, self.full_surfaces, strict=True):
      query.surface_grid.copy_(full * scale)
    self.camera.recreate(model, self.sim.expanded_fields)
    self.sim.create_graph()
