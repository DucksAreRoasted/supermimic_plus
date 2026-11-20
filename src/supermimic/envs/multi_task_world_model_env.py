"""Multi-task differentiable world model env for concurrent encoder training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.representations import root_waypoints
from supermimic.tasks.task_spec import build_actor_obs_td

if TYPE_CHECKING:
  from supermimic.tasks.walk_terrain import TerrainObservation
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation
  from supermimic.tasks.task_spec import TaskSpec


class MultiTaskWorldModelEnv(WorldModelEnv):
  """WM env for motion tracking with per-slot goal modalities.

  Reward is motion-tracking for every slot; tasks differ only in what
  ``goal_<name>`` key the actor's encoder consumes. ``get_observations``
  emits all per-task goal keys full-batch and the actor slices its own
  slot range. The obs-history ring (history_length > 1) lives in the base.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    task_specs: list[TaskSpec],
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
    terrain_query: TerrainObservation | None = None,
    actor_sees_terrain: bool = False,
    use_root_waypoints: bool = False,
  ) -> None:
    self.task_specs = task_specs
    self.terrain_query = terrain_query
    self.actor_sees_terrain = actor_sees_terrain
    self.use_root_waypoints = use_root_waypoints
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      env=env,
      batch_size=batch_size,
      replay_buffer=replay_buffer,
      history_length=history_length,
      termination_softness=termination_softness,
      device=device,
    )

  def _build_task_goals(self) -> None:
    # The anchor task's goal is the actor goal already built in _update_goal; reuse it.
    anchor, *rest = self.task_specs
    self._task_goal_obs: dict[str, torch.Tensor] = {anchor.name: self._actor_goal_obs}
    for ts in rest:
      self._task_goal_obs[ts.name] = ts.extract_training_goal(
        self.motion, self.motion_times, self.aux_state
      ).clone()

  def get_observations(self) -> TensorDict:
    obs = build_actor_obs_td(
      self._task_goal_obs, self._sensor_history(), critic=self._critic_obs()
    )
    if self.terrain_query is not None:
      patch = self.terrain_query.features(self.aux_state)
      obs["critic"] = torch.cat([obs["critic"], patch], dim=-1)
      if self.actor_sees_terrain:
        obs["terrain"] = patch
    if self.use_root_waypoints:
      obs["root_waypoints"] = root_waypoints(self._wm_goal_obs)
    return obs
