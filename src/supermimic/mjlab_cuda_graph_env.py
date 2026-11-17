"""A capture-oriented, fixed-shape path over the mjlab G1 environment.

This adapter intentionally does not replace ``ManagerBasedRlEnv.step``. It
keeps Python managers and TensorDict construction outside the graph and exposes
only device tensors to the capture body, following BlinkMimic's step_core /
reset_masked split.
"""

from __future__ import annotations

import mujoco_warp as mjwarp
import torch
import warp as wp


class DeviceStepAdapter:
  def __init__(self, env):
    self.env = env
    self.device = torch.device(env.device)
    self.num_envs = env.num_envs
    self.action_dim = env.action_manager.total_action_dim
    self.action = torch.zeros(
      (self.num_envs, self.action_dim), device=self.device
    )
    self.reset_mask = torch.zeros(
      (self.num_envs,), dtype=torch.bool, device=self.device
    )

  def step_core_device(self, action: torch.Tensor):
    """Run one fixed-shape action/physics/reward step.

    The return value is a tuple of stable-shape tensors, so it can be captured
    without Python ids, TensorDicts, or manager output dictionaries.
    """
    self.env.action_manager.process_action(action)
    for _ in range(self.env.cfg.decimation):
      self.env.action_manager.apply_action()
      self.env.scene.write_data_to_sim()
      self.env.sim.step()
      self.env.scene.update(dt=self.env.physics_dt)
    self.env.episode_length_buf.add_(1)
    self.env.common_step_counter += 1
    self.env.reset_buf.copy_(self.env.termination_manager.compute())
    self.env.reward_buf.copy_(
      self.env.reward_manager.compute(dt=self.env.step_dt)
    )
    return (
      self.env.reward_buf,
      self.env.termination_manager.terminated,
      self.env.termination_manager.time_outs,
      self.env.episode_length_buf,
    )

  def reset_masked(self, mask: torch.Tensor) -> None:
    """Reset simulator bookkeeping using a fixed device-side mask."""
    self.reset_mask.copy_(mask)
    sim = self.env.sim
    sim._reset_mask.copy_(self.reset_mask)
    with wp.ScopedDevice(sim.wp_device):
      if sim.use_cuda_graph and sim.reset_graph is not None:
        wp.capture_launch(sim.reset_graph)
      else:
        mjwarp.reset_data(sim.wp_model, sim.wp_data, reset=sim._reset_mask_wp)
    self.env.episode_length_buf.masked_fill_(mask, 0)
    self.env.reset_buf.masked_fill_(mask, False)
    self.env.reset_terminated.masked_fill_(mask, False)
    self.env.reset_time_outs.masked_fill_(mask, False)
    self.env.action_manager._action.masked_fill_(mask[:, None], 0.0)
    self.env.action_manager._prev_action.masked_fill_(mask[:, None], 0.0)
    self.env.action_manager._prev_prev_action.masked_fill_(mask[:, None], 0.0)

