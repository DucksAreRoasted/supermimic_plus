"""Imagined actor-critic updates with optional simulator value calibration."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import _optimize_step, _step_with_grads
from supermimic.learners.policy_learner import ActorCriticPolicyLearner, gae_returns


def calibration_targets(values, last_value, rewards, resets, censored, gamma, lam):
  # Cut every reset. A censored time-limit transition has no observed successor
  # before auto-reset: its target is V(s_t), yielding zero advantage there.
  # Earlier transitions thus bootstrap at their own last valid observation.
  labels = torch.where(censored, values.squeeze(-1), rewards)
  return gae_returns(
    values, last_value, labels.unsqueeze(-1), resets.to(values.dtype).unsqueeze(-1), gamma, lam
  )


def masked_sim_mean(x, valid, count):
  return torch.where(valid, x, 0.0).sum() / count


def _critic_transition(obs, reward, done):
  return TensorDict(
    {"critic": obs["critic"].detach(), "reward": reward.detach(), "done": done.detach()},
    batch_size=reward.shape,
  )


class SimulatorCalibratedActorCriticLearner(ActorCriticPolicyLearner):
  def __init__(
    self,
    *args,
    sim_lambda: float = 1.0,
    sim_calibration_weight: float = 0.0,
    sim_calibration_horizon: int = 32,
    sim_calibration_batch_size: int | None = None,
    sim_critic_input_dim: int | None = None,
    actor_microbatch_size: int | None = None,
    **kwargs,
  ):
    self.sim_calibration_epochs = int(kwargs.pop("sim_calibration_epochs", 1))
    if self.sim_calibration_epochs < 1:
      raise ValueError("sim_calibration_epochs must be at least 1")
    super().__init__(*args, **kwargs)
    self.sim_calibration_weight = sim_calibration_weight
    self.actor_microbatch_size = actor_microbatch_size
    if actor_microbatch_size is not None and actor_microbatch_size < self.cfg.batch_size:
      # Replaying a single graph for the full set of 32-step visual microbatches
      # crashes on the L40. Keep this update eager while the WM uses its graph.
      self._captured_update = self._update_from
    self.sim_lambda = sim_lambda
    self.sim_calibration_horizon = int(sim_calibration_horizon)
    self._simulator_obs: torch.Tensor | None = None
    self._simulator_rewards: torch.Tensor | None = None
    self._simulator_resets: torch.Tensor | None = None
    self._simulator_censored: torch.Tensor | None = None
    self._simulator_enabled = torch.zeros((), device=self.actor_params[0].device)
    self._actor_beta_v = torch.ones_like(self._simulator_enabled)
    self._sim_calibration_epochs_tensor = torch.full_like(
      self._simulator_enabled, self.sim_calibration_epochs
    )
    if sim_calibration_batch_size is not None and sim_critic_input_dim is not None:
      self.initialize_simulator_batch(sim_calibration_batch_size, sim_critic_input_dim)
    self._actor_updates = 0

  def _beta_v(self) -> float:
    # Keep the original Actor objective intact: the calibrated Critic should
    # improve the terminal value estimate without attenuating its bootstrap.
    return 1.0

  def _bootstrap(self, obs, survival, term_prob):
    value = super()._bootstrap(obs, survival, term_prob)
    return self._beta_v() * value

  def _update_critic_before_actor(self, start):
    rollout = []
    with torch.no_grad():
      obs = self.wm_env.reset(start)
      for _ in range(self.cfg.horizon):
        actions = self._rollout_actor(obs)
        next_obs, reward, term_prob, _ = self.wm_env.step(actions)
        rollout.append(_critic_transition(obs, reward, term_prob))
        obs = next_obs
    return self._fit_critic(rollout, obs)

  def _train_batch(self, start):
    shared_rollouts = [] if self.sim_calibration_weight == 0 else None
    if shared_rollouts is None:
      critic_logs = self._update_critic_before_actor(start)

    chunks = (start,) if self.actor_microbatch_size is None else start.split(self.actor_microbatch_size, dim=0)
    for optimizer in self.actor_optimizers:
      optimizer.zero_grad(set_to_none=True)
    actor_logs = {}
    for chunk in chunks:
      weight = len(chunk) / len(start)
      loss, logs = self._actor_objective(chunk, shared_rollouts)
      (loss * weight).backward()
      for key, value in logs.items():
        actor_logs[key] = actor_logs.get(key, 0.0) + weight * value
    grad_norm = _step_with_grads(self.actor_optimizers, self.actor_params, 1.0, self.cfg.grad_clip_value)

    if shared_rollouts is not None:
      # Join environments, not time: GAE must bootstrap each trajectory from
      # its own terminal state. Fit once on the full effective batch.
      steps, terminal = shared_rollouts[0]
      if len(shared_rollouts) > 1:
        steps = torch.cat([s for s, _ in shared_rollouts], dim=1)
        terminal = torch.cat([v for _, v in shared_rollouts], dim=0)
      critic_logs = self._fit_critic_batch(steps, terminal)
    self._actor_updates += 1
    return {**actor_logs, "grad_norm": grad_norm, "actor_beta_v": self._actor_beta_v, **critic_logs}

  def _actor_objective(self, start, shared_rollouts=None):
    cfg = self.cfg
    obs = self.wm_env.reset(start)
    reward_ret, aux_loss, survival = 0.0, 0.0, 1.0
    critic_steps = []
    for t in range(cfg.horizon):
      actions = self._rollout_actor(obs)
      next_obs, reward, term_prob, _ = self.wm_env.step(actions)
      if shared_rollouts is not None:
        critic_steps.append(_critic_transition(obs, reward, term_prob))
      obs = next_obs
      reward_ret = reward_ret + cfg.gamma**t * survival * reward
      aux_loss = aux_loss + getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
      if cfg.survival_masking:
        survival = survival * (1 - term_prob)
    if shared_rollouts is not None:
      shared_rollouts.append((torch.stack(critic_steps), obs["critic"].detach()))
    terminal = self._bootstrap(obs, survival, term_prob)
    total_return = reward_ret + terminal
    loss = (-total_return + aux_loss).mean()
    return loss, {
      "return": total_return.mean().detach(),
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
    resets: torch.Tensor,
    censored: torch.Tensor,
  ) -> None:
    """Copy one fixed simulator window into the graph-owned device buffers."""
    device = self.actor_params[0].device
    if self._simulator_obs is None:
      self.initialize_simulator_batch(critic_obs.shape[1], critic_obs.shape[2])
    assert self._simulator_obs is not None
    assert self._simulator_rewards is not None
    assert self._simulator_resets is not None
    self._simulator_obs.copy_(critic_obs.detach().to(device))
    self._simulator_rewards.copy_(rewards.detach().to(device))
    assert self._simulator_censored is not None
    self._simulator_resets.copy_(resets.detach())
    self._simulator_censored.copy_(censored.detach())
    self._simulator_enabled.fill_(1.0)

  def initialize_simulator_batch(self, batch_size: int, critic_input_dim: int) -> None:
    """Allocate fixed-shape simulator targets before CUDA Graph capture."""
    device = self.actor_params[0].device
    horizon = self.sim_calibration_horizon
    self._simulator_obs = torch.zeros(
      horizon + 1, batch_size, critic_input_dim, device=device
    )
    self._simulator_rewards = torch.zeros(horizon, batch_size, device=device)
    self._simulator_resets = torch.zeros(horizon, batch_size, dtype=torch.bool, device=device)
    self._simulator_censored = torch.zeros_like(self._simulator_resets)
    self._simulator_enabled.zero_()

  def clear_simulator_batch(self) -> None:
    """Disable the previous simulator window without changing its storage."""
    self._simulator_enabled.zero_()

  def _graph_state(self) -> list[torch.Tensor]:
    """Include fixed calibration buffers in CUDA Graph identity checks."""
    simulator_state = [
      buffer
      for buffer in (
        self._simulator_obs,
        self._simulator_rewards,
        self._simulator_resets,
        self._simulator_censored,
        self._simulator_enabled,
        self._actor_beta_v,
        self._sim_calibration_epochs_tensor,
      )
      if buffer is not None
    ]
    return super()._graph_state() + simulator_state

  def _fit_critic(self, rollout, last_obs) -> dict[str, torch.Tensor]:
    return self._fit_critic_batch(torch.stack(rollout), last_obs["critic"])

  def _fit_critic_batch(self, steps, terminal_obs) -> dict[str, torch.Tensor]:
    sim_weight = self.sim_calibration_weight
    with torch.no_grad():
      values = self.critic(steps["critic"])
      last_value = self.critic(terminal_obs)
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
    imagined_loss = torch.nn.functional.mse_loss(values.flatten(0, 1), imagined_targets)

    sim_loss = torch.zeros((), device=imagined_loss.device)
    sim_return = torch.zeros((), device=imagined_loss.device)
    if sim_weight > 0:
      assert self._simulator_obs is not None
      assert self._simulator_rewards is not None
      assert self._simulator_resets is not None
      sim_obs = self._simulator_obs
      sim_rewards = self._simulator_rewards
      assert self._simulator_censored is not None
      censored = self._simulator_censored
      valid = ~censored.flatten().unsqueeze(-1)
      valid_count = valid.sum().clamp_min(1)
      with torch.no_grad():
        sim_values = self.critic(sim_obs[:-1])
        sim_last_value = self.critic(sim_obs[-1])
        sim_returns = calibration_targets(
          sim_values,
          sim_last_value,
          sim_rewards,
          self._simulator_resets,
          censored,
          self.cfg.gamma,
          self.sim_lambda,
        ).clamp(max=self._v_max)
      sim_targets = sim_returns.flatten(0, 1).detach()
      sim_loss = masked_sim_mean(
        (sim_values.flatten(0, 1) - sim_targets).square(), valid, valid_count
      )
      sim_return = masked_sim_mean(sim_targets, valid, valid_count)

    # Keep both return targets fixed while reusing this real simulator window.
    # Additional epochs improve value regression without collecting more data.
    grad_norm = torch.zeros((), device=imagined_loss.device)
    for _ in range(self.sim_calibration_epochs):
      imagined_prediction = self.critic(imagined_obs)
      sim_prediction = (
        self.critic(self._simulator_obs[:-1].flatten(0, 1))
        if sim_weight > 0
        else None
      )
      imagined_epoch_loss = torch.nn.functional.mse_loss(
        imagined_prediction, imagined_targets
      )
      sim_epoch_loss = (
        masked_sim_mean((sim_prediction - sim_targets).square(), valid, valid_count)
        if sim_prediction is not None
        else torch.zeros((), device=imagined_loss.device)
      )
      total_loss = imagined_epoch_loss + (
        self._simulator_enabled * sim_weight * sim_epoch_loss
      )
      grad_norm = _optimize_step(
        self.critic_optimizers,
        total_loss,
        self.critic.parameters(),
        max_norm=1.0,
      )
    window_stats = {}
    if sim_weight > 0:
      window_stats = {
        "sim_reward_time_delta": (sim_rewards[1:] - sim_rewards[:-1]).abs().sum()
        / max(1, sim_rewards[1:].numel()) * self._simulator_enabled,
        "sim_reset_fraction": self._simulator_resets.float().mean() * self._simulator_enabled,
        "sim_censored_fraction": censored.float().mean() * self._simulator_enabled,
      }
    return {
      **window_stats,
      "value_function": imagined_loss.detach(),
      "imagined_value_loss": imagined_loss.detach(),
      "sim_calibration_loss": self._simulator_enabled * sim_loss.detach(),
      "critic_total_loss": total_loss.detach(),
      "sim_calibration_return": (
        self._simulator_enabled * sim_return.detach()
      ),
      "critic_grad_norm": grad_norm,
      "sim_calibration_epochs": self._sim_calibration_epochs_tensor,
    }
