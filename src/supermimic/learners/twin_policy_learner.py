"""Experimental twin critic; the official learner stays in policy_learner.py."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.learner_utils import (
  _optimize_step,
  build_optimizers,
  graph_state,
)
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  gae_returns,
)


class TwinActorCriticPolicyLearner(BpttPolicyLearner):
  """Actor-critic variant that adds a critic bootstrap and GAE critic update."""

  def __init__(
    self,
    policy: nn.Module,
    critic1: nn.Module,
    critic2: nn.Module,
    wm_env: WorldModelEnv,
    cfg: ActorCriticPolicyLearnerCfg,
    device: str = "cpu",
  ):
    super().__init__(policy, wm_env, cfg, device)
    self.critics = nn.ModuleList([critic1, critic2])
    self._assert_independent_critics()
    self.critic_optimizers = build_optimizers(
      [self.critics], self.cfg.lr, device=device
    )

  @property
  def critic(self) -> nn.Module:
    """Primary critic kept as a compatibility view for existing callers."""
    return self.critics[0]

  def _values(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.stack([module(obs) for module in self.critics], dim=0)

  def _assert_independent_critics(self) -> None:
    """Reject accidental parameter storage sharing at the construction seam."""
    storages = [
      {parameter.untyped_storage().data_ptr() for parameter in critic.parameters()}
      for critic in self.critics
    ]
    if storages[0] & storages[1]:
      raise ValueError("critic1 and critic2 must not share parameter storage")

  def _bootstrap(
    self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor
  ) -> torch.Tensor:
    self.critics.requires_grad_(False)
    value = self._values(obs["critic"]).amin(dim=0).squeeze(-1)
    alive = survival if self.cfg.survival_masking else 1.0 - term_prob
    return self.cfg.gamma**self.cfg.horizon * value * alive

  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    steps = torch.stack(rollout)
    with torch.no_grad():
      twin_values = self._values(steps["critic"])
      last_value = self._values(last_obs["critic"])
      values = twin_values.amin(dim=0)
      last_value = last_value.amin(dim=0)
    returns = gae_returns(
      values,
      last_value,
      steps["reward"].unsqueeze(-1),
      steps["done"].unsqueeze(-1),
      self.cfg.gamma,
      self.cfg.lam,
    )
    # Snapshot diagnostics before fitting, using the same conservative values
    # and soft termination discounts as GAE. Keep scalars on device for logging.
    with torch.no_grad():
      next_values = torch.cat((values[1:], last_value.unsqueeze(0)), dim=0)
      td_error = (
        steps["reward"].unsqueeze(-1)
        + self.cfg.gamma * (1 - steps["done"].unsqueeze(-1)) * next_values
        - values
      )
      # Fixed-shape sort/interpolation avoids quantile's host-side checks during
      # CUDA graph capture; these are linearly interpolated 1st/99th percentiles.
      sorted_targets = returns.flatten().sort().values
      percentiles = {}
      for name, q in (("01", 0.01), ("99", 0.99)):
        position = (sorted_targets.numel() - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        percentiles[f"target_percentile_{name}"] = torch.lerp(
          sorted_targets[lo], sorted_targets[hi], position - lo
        )
      stats = {
        "value_mean": values.mean(),
        "value_std": values.std(correction=0),
        "value_min": values.min(),
        "value_max": values.max(),
        "value_1_mean": twin_values[0].mean(),
        "critic_td_error": td_error.mean(),
        "critic_td_error_abs": td_error.abs().mean(),
        "critic_td_error_rms": td_error.square().mean().sqrt(),
        "advantage_std": (returns - values).std(correction=0),
        "target_mean": returns.mean(),
        "target_std": returns.std(correction=0),
        **percentiles,
      }
      difference = twin_values[0] - twin_values[1]
      stats.update(
        value_2_mean=twin_values[1].mean(),
        critic_disagreement_mean=difference.mean(),
        critic_disagreement_var=difference.var(correction=0),
        critic_disagreement_abs=difference.abs().mean(),
      )
      if self.cfg.gamma < 1 and math.isfinite(self.wm_env.max_step_reward):
        v_max = self.wm_env.max_step_reward / (1 - self.cfg.gamma)
        stats["value_above_bound_fraction"] = (twin_values > v_max).float().mean()
    obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
    perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
    self.critics.requires_grad_(True)
    losses = []
    for _ in range(5):
      for idx in perm.view(4, -1):
        prediction = self._values(obs[idx])
        target = returns[idx].unsqueeze(0).expand_as(prediction)
        loss = torch.nn.functional.huber_loss(prediction, target)
        _optimize_step(
          self.critic_optimizers, loss, self.critics.parameters(), max_norm=1.0
        )
        losses.append(loss.detach())
    return {"value_function": torch.stack(losses).mean(), **stats}

  def _graph_state(self) -> list[torch.Tensor]:
    return super()._graph_state() + graph_state([self.critics], self.critic_optimizers)
