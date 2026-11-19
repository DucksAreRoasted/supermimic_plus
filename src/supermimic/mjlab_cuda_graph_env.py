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
    self.reset_buf = torch.zeros_like(self.reset_mask)
    self.reset_terminated = torch.zeros_like(self.reset_mask)
    self.reset_time_outs = torch.zeros_like(self.reset_mask)
    self.reward_buf = torch.zeros(
      (self.num_envs,), device=self.device, dtype=torch.float32
    )

  def step_core_device(self, action: torch.Tensor):
    """Run one fixed-shape action/physics/reward step.

    The return value is a tuple of stable-shape tensors, so it can be captured
    without Python ids, TensorDicts, or manager output dictionaries.
    """
    self.env.action_manager.process_action(action)
    for _ in range(self.env.cfg.decimation):
      self.env._sim_step_counter += 1
      self.env.action_manager.apply_action()
      self.env.scene.write_data_to_sim()
      self.env.sim.step()
      self.env.scene.update(dt=self.env.physics_dt)
      self.env.metrics_manager.compute_substep()
    self.env.episode_length_buf.add_(1)
    self.env.common_step_counter += 1
    self.reset_buf.copy_(self.env.termination_manager.compute())
    self.reset_terminated.copy_(self.env.termination_manager.terminated)
    self.reset_time_outs.copy_(self.env.termination_manager.time_outs)
    self.reward_buf.copy_(
      self.env.reward_manager.compute(dt=self.env.step_dt)
    )
    return (
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.env.episode_length_buf,
    )

  def post_step(self, terminated: torch.Tensor, time_outs: torch.Tensor) -> None:
    """Run the Python-side reset and observation hooks omitted from the graph.

    ManagerBasedRlEnv.step performs these operations after reward computation.
    Keeping them outside the graph preserves command sampling, motion timelines,
    observation history, and event callbacks while the expensive physics step
    remains captured.
    """
    env = self.env
    reset_mask = terminated | time_outs
    # These attributes are created lazily by ManagerBasedRlEnv.step. Assign the
    # stable graph-owned tensors here after capture, rather than reading them
    # during graph construction.
    env.reset_buf = reset_mask
    env.reset_terminated = terminated
    env.reset_time_outs = time_outs
    env.reward_buf = self.reward_buf
    reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
    if "step" in env.event_manager.available_modes:
      env.event_manager.apply(mode="step", dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
      env.event_manager.apply(mode="interval", dt=env.step_dt)
    if len(reset_ids) > 0:
      env.recorder_manager.record_pre_reset(reset_ids)
      env._reset_idx(reset_ids)
      env.scene.write_data_to_sim(env_ids=reset_ids)
    env.sim.forward()
    command_dt = env._command_dt
    command_dt.fill_(env.step_dt)
    if len(reset_ids) > 0:
      command_dt[reset_ids] = 0.0
    env.command_manager.compute(dt=command_dt)
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    env.metrics_manager.compute()
    if len(reset_ids) > 0:
      env.recorder_manager.record_post_reset(reset_ids)
    env.recorder_manager.record_post_step()


class CapturedDeviceStep:
  """CUDA-graph wrapper around :class:`DeviceStepAdapter`."""

  def __init__(self, env):
    self.adapter = DeviceStepAdapter(env)
    self.env = env
    self.action = self.adapter.action
    self._graph = torch.cuda.CUDAGraph()
    self._warmup_step()
    env.reset()
    stream = torch.cuda.Stream(device=self.action.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(self._graph, stream=stream):
      self._outputs = self._capture_body(self.action)
    torch.cuda.current_stream().wait_stream(stream)

  def _capture_body(self, action: torch.Tensor):
    return self.adapter.step_core_device(action)

  def _warmup_step(self) -> None:
    for _ in range(2):
      self._capture_body(self.action)
    torch.cuda.synchronize(self.action.device)

  def __call__(self, action: torch.Tensor):
    self.action.copy_(action)
    self._graph.replay()
    outputs = tuple(x.clone() for x in self._outputs)
    self.adapter.post_step(outputs[1], outputs[2])
    return outputs

