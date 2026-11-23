"""Place the G1 motion reference pose above the local walking surface at reset."""

from dataclasses import fields

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
