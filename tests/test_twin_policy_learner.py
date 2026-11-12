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

  # Conservative V=1; one-step TD errors are 0.25 and 0.5.
  # GAE targets are 1.375 and 1.5, each repeated for two environments.
  expected = {
    "value_mean": 1.0,
    "value_std": 0.0,
    "value_min": 1.0,
    "value_max": 1.0,
    "value_1_mean": 1.0,
    "value_2_mean": 3.0,
    "critic_td_error": 0.375,
    "critic_td_error_abs": 0.375,
    "critic_td_error_rms": (0.15625) ** 0.5,
    "critic_disagreement_mean": -2.0,
    "critic_disagreement_var": 0.0,
    "critic_disagreement_abs": 2.0,
    "advantage_std": 0.0625,
    "target_mean": 1.4375,
    "target_std": 0.0625,
    "target_percentile_01": 1.375,
    "target_percentile_99": 1.5,
    "value_above_bound_fraction": 0.5,
  }
  for name, value in expected.items():
    actual = stats[f"Policy/{name}"]
    assert actual.ndim == 0
    assert not actual.requires_grad
    assert actual.item() == pytest.approx(value)


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
