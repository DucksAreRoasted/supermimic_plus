"""Official actor-critic learner with an extra real-simulator value update."""

from __future__ import annotations

import torch

from supermimic.learners.learner_utils import _optimize_step
from supermimic.learners.policy_learner import ActorCriticPolicyLearner


class SimulatorCalibratedActorCriticLearner(ActorCriticPolicyLearner):
  @torch.no_grad()
  def simulator_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
      running = rewards[t] + self.cfg.gamma * (1 - dones[t]) * running
      returns[t] = running
    return returns

  def calibrate_from_simulator(
    self, critic_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, weight: float
  ) -> dict[str, torch.Tensor]:
    if weight <= 0:
      return {}
    targets = self.simulator_returns(rewards, dones)[0].detach()
    prediction = self.critic(critic_obs[0].detach())
    loss = torch.nn.functional.mse_loss(prediction, targets.unsqueeze(-1))
    _optimize_step(
      self.critic_optimizers,
      weight * loss,
      self.critic.parameters(),
      max_norm=1.0,
    )
    return {"sim_calibration_loss": loss.detach(), "sim_calibration_return": targets.mean()}
