"""Official actor-critic learner with an extra real-simulator value update."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import _optimize_step
from supermimic.learners.policy_learner import ActorCriticPolicyLearner, gae_returns


class SimulatorCalibratedActorCriticLearner(ActorCriticPolicyLearner):
  def __init__(self, *args, sim_lambda: float = 1.0, **kwargs):
    super().__init__(*args, **kwargs)
    self.sim_lambda = sim_lambda
    self._simulator_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    self._actor_updates = 0
    # Separate gradient probes retain the actor graph, which is incompatible
    # with the donated buffers used by CapturedUpdate/CUDA Graph replay.
    self._captured_update = self._update_from

  def _beta_v(self) -> float:
    return min(1.0, 0.25 + 0.75 * self._actor_updates / 2000.0)

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
    grads_r = torch.autograd.grad(reward_ret.mean(), self.actor_params, retain_graph=True, allow_unused=True)
    grads_v = torch.autograd.grad(terminal.mean() if isinstance(terminal, torch.Tensor) else total_return.mean(), self.actor_params, retain_graph=True, allow_unused=True)
    flat_r = [g.reshape(-1) for g in grads_r if g is not None]
    flat_v = [g.reshape(-1) for g in grads_v if g is not None]
    if flat_r and flat_v:
      cosine = torch.nn.functional.cosine_similarity(
        torch.cat(flat_r), torch.cat(flat_v), dim=0
      )
    else:
      cosine = torch.zeros((), device=total_return.device)
    loss = (-total_return + aux_loss).mean()
    grad_norm = _optimize_step(self.actor_optimizers, loss, self.actor_params, 1.0, cfg.grad_clip_value)
    self._actor_updates += 1
    return {
      "return": total_return.mean().detach(),
      "grad_norm": grad_norm,
      "actor_beta_v": torch.tensor(self._beta_v(), device=total_return.device),
      "actor_gradient_cosine": cosine.detach(),
      **critic_logs,
      **getattr(self.actor, "auxiliary_stats", lambda: {})(),
    }

  def set_simulator_batch(
    self,
    critic_obs: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
  ) -> None:
    """Set one fixed H+1-state simulator window for the next critic update."""
    self._simulator_batch = (critic_obs.detach(), rewards.detach(), terminated.detach())

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

    total_loss = imagined_loss + sim_weight * sim_loss
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
    }
