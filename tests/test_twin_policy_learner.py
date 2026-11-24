import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
  gae_returns,
)
from supermimic.learners.twin_policy_learner import (
  TwinActorCriticPolicyLearner as ActorCriticPolicyLearner,
)
from supermimic.learners.sim_calibrated_learner import SimulatorCalibratedActorCriticLearner

NUM_ENVS = 2


class _DummyActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(()))

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return obs["actor"] * self.weight


class _DummyRepresentation(nn.Module):
  dt = 1.0


class _TwoStepEnv:
  """Reward 1 per step; the first step terminates with probability 0.5."""

  num_envs = NUM_ENVS
  num_actions = 1
  max_step_reward = float("inf")

  def __init__(self):
    self.dynamics = nn.Linear(1, 1)
    self.representation = _DummyRepresentation()
    self.step_idx = 0

  def sample_initial_states(self) -> TensorDict:
    return TensorDict({}, batch_size=[NUM_ENVS])

  def reset(self, batch: TensorDict) -> TensorDict:
    del batch
    self.step_idx = 0
    return self._obs()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    rewards = 1.0 + actions.squeeze(-1)
    term_prob = torch.full((NUM_ENVS,), 0.5 if self.step_idx == 0 else 0.0)
    self.step_idx += 1
    return self._obs(), rewards, term_prob, {}

  def _obs(self) -> TensorDict:
    obs = {"actor": torch.zeros(NUM_ENVS, 1), "critic": torch.zeros(NUM_ENVS, 1)}
    return TensorDict(obs, batch_size=[NUM_ENVS])


class _ConstantCritic(nn.Module):
  def __init__(self, value: float):
    super().__init__()
    self.value = nn.Parameter(torch.tensor(value))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return obs * 0.0 + self.value


def test_survival_masking_counts_current_transition_reward_before_done():
  learner = BpttPolicyLearner(
    _DummyActor(),
    _TwoStepEnv(),
    BpttPolicyLearnerCfg(horizon=2, gamma=1.0, survival_masking=True),
  )

  stats = learner.update()

  # Return 1 + (1 - 0.5) * 1, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx(1.5 / 2)


def test_actor_critic_bootstraps_the_surviving_last_state():
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _ConstantCritic(1.0),
    _TwoStepEnv(),
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.99, survival_masking=True),
  )

  stats = learner.update()

  # Return 1 + 0.99 * 0.5 * 1, plus the value 1 discounted by 0.99^2 and by
  # survival 0.5, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx((1.495 + 0.49005) / 2)
  assert torch.isfinite(stats["Policy/value_function"])


def test_gae_returns_discount_by_soft_done():
  returns = gae_returns(
    values=torch.zeros(2, 1),
    last_value=torch.zeros(1),
    rewards=torch.ones(2, 1),
    dones=torch.tensor([[0.5], [0.0]]),
    gamma=0.99,
    lam=1.0,
  )

  assert torch.allclose(returns, torch.tensor([[1.495], [1.0]]))


def test_simulator_calibration_returns_stop_at_done():
  learner = SimulatorCalibratedActorCriticLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _TwoStepEnv(),
    ActorCriticPolicyLearnerCfg(gamma=0.5),
  )
  rewards = torch.tensor([[1.0], [2.0], [3.0]])
  dones = torch.tensor([[0.0], [1.0], [0.0]])
  assert torch.allclose(
    learner.simulator_returns(rewards, dones),
    torch.tensor([[2.0], [2.0], [3.0]]),
  )


def test_critic_diagnostics_use_prefit_values_and_soft_done():
  env = _TwoStepEnv()
  env.max_step_reward = 0.5
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _ConstantCritic(3.0),
    env,
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.5, lam=1.0),
  )
  with torch.no_grad():
    learner.critics[1].value.fill_(3.0)

  stats = learner.update()

  # Each critic now builds its own GAE target; actor aggregation defaults to mean.
  assert stats["Policy/value_mean"].item() == pytest.approx(2.0)
  assert stats["Policy/value_1_mean"].item() == pytest.approx(1.0)
  assert stats["Policy/value_2_mean"].item() == pytest.approx(3.0)
  assert stats["Policy/critic_disagreement_abs"].item() == pytest.approx(2.0)
  assert stats["Policy/target_mean"].item() >= 1.0
  assert all(torch.isfinite(value) for value in stats.values())


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_critic_diagnostics_without_finite_value_bound(gamma):
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _ConstantCritic(1.0),
    _TwoStepEnv(),
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=gamma, num_batches=2),
  )
  stats = learner.update()
  assert "Policy/value_above_bound_fraction" not in stats
  assert stats["Policy/critic_disagreement_abs"].item() == 0.0
  assert all(torch.isfinite(value) for value in stats.values())


@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_sim_calibration_weight_selects_rollout_and_update_order(weight):
  class Actor(_DummyActor):
    def forward(self, obs):
      return torch.ones_like(obs["actor"]) * self.weight

  actor = Actor()
  env = _TwoStepEnv()
  learner = SimulatorCalibratedActorCriticLearner(
    actor, _ConstantCritic(0.2), env, ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.99),
    sim_calibration_weight=weight, sim_calibration_horizon=2,
  )
  if weight:
    resets = torch.zeros(2, 2, dtype=torch.bool)
    learner.set_simulator_batch(torch.zeros(3, 2, 1), torch.ones(2, 2), resets, resets)
  grad_enabled = []
  events = []
  actor.register_forward_hook(lambda *_: grad_enabled.append(torch.is_grad_enabled()))
  for name, optimizer in (("actor", learner.actor_optimizers[0]), ("critic", learner.critic_optimizers[0])):
    optimizer.register_step_post_hook(lambda *_, label=name: events.append(label))
  before = actor.weight.detach().clone()
  result = learner.update()
  assert grad_enabled == ([True, True] if weight == 0 else [False, False, True, True])
  assert events == (["actor", "critic"] if weight == 0 else ["critic", "actor"])
  if weight == 0:
    assert result["Policy/reward"].item() == pytest.approx((1 + .99 * .5 + .99**2 * .5 * .2) / 2)
  assert bool(result["Policy/sim_calibration_loss"] > 0) == bool(weight)
  assert result["Policy/grad_norm"] > 0
  assert not torch.equal(actor.weight, before)


def test_actor_microbatch_matches_one_full_update_with_unequal_chunks():
  class Env(_TwoStepEnv):
    def sample_initial_states(self):
      return TensorDict({"initial": torch.linspace(-1, 3, 5)[:, None]}, batch_size=[5])

    def reset(self, batch):
      self.state = batch["initial"]
      return self._obs()

    def _obs(self):
      return TensorDict({"actor": self.state, "critic": self.state}, batch_size=[len(self.state)])

    def step(self, actions):
      self.state = self.state + .1 * actions
      return self._obs(), (actions - .3).squeeze(-1), self.state[:, 0].sigmoid() * .1, {}

  def experiment(chunk):
    actor = _DummyActor()
    env = Env()
    learner = SimulatorCalibratedActorCriticLearner(
      actor, _ConstantCritic(.2), env, ActorCriticPolicyLearnerCfg(horizon=32, gamma=.99, batch_size=5),
      actor_microbatch_size=chunk,
    )
    if chunk is not None and chunk < learner.cfg.batch_size:
      assert learner._captured_update == learner._update_from
    imagined_rows = []
    actor.register_forward_hook(lambda _, args, out: imagined_rows.append(len(args[0])))
    fit = learner._fit_critic_batch
    batches = []
    def checked_fit(steps, terminal):
      assert steps.batch_size == (32, 5) and terminal.shape == (5, 1)
      assert not terminal.requires_grad and all(not x.requires_grad for x in steps.values())
      batches.append((steps.clone(), terminal.clone()))
      return fit(steps, terminal)
    learner._fit_critic_batch = checked_fit
    logs = learner.update()
    assert sum(imagined_rows) == 5 * 32
    assert len(batches) == 1
    return actor.weight.detach(), learner.critic.value.detach(), logs, learner._actor_updates, batches[0]

  w, v, logs, count, (steps, terminal) = experiment(None)
  split_w, split_v, split_logs, split_count, (split_steps, split_terminal) = experiment(2)
  for key in steps.keys():
    torch.testing.assert_close(steps[key], split_steps[key])
  torch.testing.assert_close(terminal, split_terminal)
  assert count == split_count == 1
  assert w.abs() > 0
  torch.testing.assert_close(w, split_w)
  torch.testing.assert_close(v, split_v)
  for key in logs:
    torch.testing.assert_close(logs[key], split_logs[key])
