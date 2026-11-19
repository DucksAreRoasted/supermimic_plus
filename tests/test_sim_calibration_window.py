"""Regression tests for the data that actually enters the calibrated Critic."""

import pytest
import torch

from supermimic.learners.sim_calibrated_learner import (
  calibration_targets,
  masked_sim_mean,
)
from supermimic.runners.sim_calibration_window import SimulatorWindow


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
  if request.param == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable")
  with torch.device(request.param):
    yield


def test_reward_manager_reused_storage_and_window_rollover(device):
  window = SimulatorWindow(32)
  reward = torch.zeros(1)
  reset = torch.zeros(1, dtype=torch.bool)
  for i in range(34):
    x = torch.full((1, 1), float(i))
    window.append(x, x + 1, reward.fill_(i + 1), reset, reset, reset)
  reward.fill_(-8)
  reset.fill_(True)
  obs, rewards, boundaries, censored = window.batch()
  torch.testing.assert_close(rewards[:, 0], torch.arange(3, 35).float())
  torch.testing.assert_close(obs[:, 0, 0], torch.arange(2, 35).float())
  assert not boundaries.any() and not censored.any()
  targets = calibration_targets(
    obs[:-1], obs[-1], rewards, boundaries, censored, 1.0, 1.0
  )
  assert targets[0, 0, 0] == sum(range(3, 35)) + 34  # still bootstraps on open tail


@pytest.mark.parametrize("lam", [0.0, 0.7, 1.0])
def test_reset_isolation_for_fall_clip_end_and_timeout(device, lam):
  # Four envs: true termination, natural end of clip, pure time limit, no reset.
  def run(new_episode):
    window = SimulatorWindow(3)
    flags = torch.tensor([True, True, True, False])
    for t in range(3):
      obs = torch.full((4, 1), [4.0, 7.0, new_episode][t])
      nxt = torch.full_like(obs, [7.0, new_episode, 3.0][t])
      flags_t = flags & (t == 1)
      r = torch.full((4,), [1.0, 2.0, new_episode][t])
      window.append(
        obs,
        nxt,
        r,
        flags_t,
        flags_t & torch.tensor([1, 0, 0, 0]).bool(),
        flags_t & torch.tensor([0, 1, 0, 0]).bool(),
      )
    obs, r, resets, censored = window.batch()
    return calibration_targets(
      obs[:-1], obs[-1], r, resets, censored, 0.9, lam
    ), censored

  base, censored = run(11.0)
  changed, _ = run(1111.0)
  torch.testing.assert_close(base[:2, :3], changed[:2, :3])
  assert not torch.equal(base[2], changed[2])
  torch.testing.assert_close(base[1, :3, 0], torch.tensor([2.0, 2.0, 7.0]))
  assert base[0, 2, 0].item() == pytest.approx(1.0 + 0.9 * 7.0)
  assert censored.nonzero().tolist() == [[1, 2]]
  if lam > 0:
    assert base[0, 3, 0] != changed[0, 3, 0]


def test_censored_transition_has_no_regression_gradient(device):
  x = torch.tensor([[3.0], [100.0]], requires_grad=True)
  valid = torch.tensor([[True], [False]])
  loss = masked_sim_mean(x.square(), valid, valid.sum().clamp_min(1))
  loss.backward()
  assert loss.item() == 9.0
  assert x.grad.flatten().tolist() == [6.0, 0.0]
  assert (
    masked_sim_mean(x.square(), ~torch.ones_like(valid), torch.tensor(1)).item() == 0.0
  )
