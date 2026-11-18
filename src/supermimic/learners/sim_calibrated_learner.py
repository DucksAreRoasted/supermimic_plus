"""Official actor-critic learner with an extra real-simulator value update."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import _optimize_step
from supermimic.learners.policy_learner import ActorCriticPolicyLearner, gae_returns


class SimulatorCalibratedActorCriticLearner(ActorCriticPolicyLearner):
  def __init__(self, *args, sim_lambda: float = 1.0, **kwargs):
    self.sim_calibration_epochs = int(kwargs.pop("sim_calibration_epochs", 1))
    if self.sim_calibration_epochs < 1:
      raise ValueError("sim_calibration_epochs must be at least 1")
    super().__init__(*args, **kwargs)
    self.sim_lambda = sim_lambda
    self._simulator_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    self._actor_updates = 0
    # The simulator batch is injected dynamically between policy updates;
    # capturing the update before/after that branch changes is unsafe.
    self._captured_update = self._update_from

  def _beta_v(self) -> float:
    # Keep the original Actor objective intact: the calibrated Critic should
    # improve the terminal value estimate without attenuating its bootstrap.
    return 1.0

  def _bootstrap(self, obs, survival, term_prob):
    value = super()._bootstrap(obs, survival, term_prob)
    return self._beta_v() * value

  def _train_batch(self, start):
    cfg = self.cfg
    obs = self.wm_env.reset(start)
    survival = 1.0
    rollout = []
    for _ in range(cfg.horizon):
      actions = self._rollout_actor(obs)
      next_obs, reward, term_prob, _ = self.wm_env.step(actions)
      rollout.append(TensorDict({"critic": obs["critic"].detach(), "reward": reward.detach(), "done": term_prob.detach()}, batch_size=reward.shape))
      if cfg.survival_masking:
        survival = survival * (1 - term_prob)
      obs = next_obs
    critic_logs = self._fit_critic(rollout, obs)
    obs = self.wm_env.reset(start)
    reward_ret, aux_loss, survival = 0.0, 0.0, 1.0
    for t in range(cfg.horizon):
      actions = self._rollout_actor(obs)
      obs, reward, term_prob, _ = self.wm_env.step(actions)
      reward_ret = reward_ret + cfg.gamma**t * survival * reward
      aux_loss = aux_loss + getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
      if cfg.survival_masking:
        survival = survival * (1 - term_prob)
    terminal = self._bootstrap(obs, survival, term_prob)
    total_return = reward_ret + terminal
    loss = (-total_return + aux_loss).mean()
    grad_norm = _optimize_step(self.actor_optimizers, loss, self.actor_params, 1.0, cfg.grad_clip_value)
    self._actor_updates += 1
    return {
      "return": total_return.mean().detach(),
      "grad_norm": grad_norm,
      "actor_beta_v": torch.tensor(self._beta_v(), device=total_return.device),
      **critic_logs,
      **getattr(self.actor, "auxiliary_stats", lambda: {})(),
    }

  def simulator_returns(
    self, rewards: torch.Tensor, dones: torch.Tensor
  ) -> torch.Tensor:
    """Compute discounted returns that stop at each simulator termination."""
    if rewards.shape != dones.shape:
      raise ValueError("rewards and dones must have the same shape")
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for index in range(rewards.shape[0] - 1, -1, -1):
      running = rewards[index] + self.cfg.gamma * (1 - dones[index]) * running
      returns[index] = running
    return returns

  def set_simulator_batch(
    self,
    critic_obs: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
  ) -> None:
    """Set one fixed H+1-state simulator window for the next critic update."""
    device = self.actor_params[0].device
    self._simulator_batch = (
      critic_obs.detach().to(device),
      rewards.detach().to(device),
      terminated.detach().to(device),
    )

  def _fit_critic(self, rollout, last_obs) -> dict[str, torch.Tensor]:
    sim_weight = getattr(self, "sim_calibration_weight", 0.0)
    steps = torch.stack(rollout)
    with torch.no_grad():
      values = self.critic(steps["critic"])
      last_value = self.critic(last_obs["critic"])
    imagined_returns = gae_returns(
      values,
      last_value,
      steps["reward"].unsqueeze(-1),
      steps["done"].unsqueeze(-1),
      self.cfg.gamma,
      self.cfg.lam,
    ).clamp(max=self._v_max)
    self.critic.requires_grad_(True)
    imagined_obs = steps["critic"].flatten(0, 1)
    imagined_targets = imagined_returns.flatten(0, 1).detach()
    imagined_loss = torch.nn.functional.mse_loss(
      self.critic(imagined_obs), imagined_targets
    )

    sim_loss = torch.zeros((), device=imagined_loss.device)
    sim_return = torch.zeros((), device=imagined_loss.device)
    if sim_weight > 0 and self._simulator_batch is not None:
      sim_obs, sim_rewards, sim_terminated = self._simulator_batch
      with torch.no_grad():
        sim_values = self.critic(sim_obs[:-1])
        sim_last_value = self.critic(sim_obs[-1])
        sim_returns = gae_returns(
          sim_values,
          sim_last_value,
          sim_rewards.unsqueeze(-1),
          sim_terminated.unsqueeze(-1),
          self.cfg.gamma,
          self.sim_lambda,
        ).clamp(max=self._v_max)
      sim_prediction = self.critic(sim_obs[:-1].flatten(0, 1))
      sim_targets = sim_returns.flatten(0, 1).detach()
      sim_loss = torch.nn.functional.mse_loss(sim_prediction, sim_targets)
      sim_return = sim_targets.mean()

    # Keep both return targets fixed while reusing this real simulator window.
    # Additional epochs improve value regression without collecting more data.
    grad_norm = torch.zeros((), device=imagined_loss.device)
    for _ in range(self.sim_calibration_epochs):
      imagined_prediction = self.critic(imagined_obs)
      sim_prediction = self.critic(sim_obs[:-1].flatten(0, 1)) if sim_weight > 0 and self._simulator_batch is not None else None
      imagined_epoch_loss = torch.nn.functional.mse_loss(
        imagined_prediction, imagined_targets
      )
      sim_epoch_loss = (
        torch.nn.functional.mse_loss(sim_prediction, sim_targets)
        if sim_prediction is not None
        else torch.zeros((), device=imagined_loss.device)
      )
      total_loss = imagined_epoch_loss + sim_weight * sim_epoch_loss
      grad_norm = _optimize_step(
        self.critic_optimizers,
        total_loss,
        self.critic.parameters(),
        max_norm=1.0,
      )
    self._simulator_batch = None
    return {
      "value_function": imagined_loss.detach(),
      "imagined_value_loss": imagined_loss.detach(),
      "sim_calibration_loss": sim_loss.detach(),
      "critic_total_loss": total_loss.detach(),
      "sim_calibration_return": sim_return.detach(),
      "critic_grad_norm": grad_norm,
      "sim_calibration_epochs": torch.tensor(
        self.sim_calibration_epochs, device=imagined_loss.device
      ),
    }
