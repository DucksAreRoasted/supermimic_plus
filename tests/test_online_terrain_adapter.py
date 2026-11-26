import torch

from supermimic.tasks.online_terrain_adapter import OnlineTerrainAdapter, adapter_inputs


def test_unknown_cells_keep_their_visibility_and_ignore_stale_heights():
  obs = {
    "goal_motion": torch.zeros(1, 680),
    "history": torch.zeros(1, 1, 68),
    "future_feet": torch.zeros(1, 2, 3, 9),
    "depth_tokens": torch.tensor([[[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 9.0, 0.0]]]),
  }
  inputs, visible = adapter_inputs(obs, torch.zeros(1, 29))
  torch.testing.assert_close(inputs[:, -4:], torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
  assert visible.item()


def test_residual_starts_at_zero_and_stays_zero_without_depth():
  model = OnlineTerrainAdapter(16, 29, 0.12)
  inputs = torch.randn(2, 16)
  visible = torch.tensor([[True], [False]])
  initial = model(inputs, visible)
  torch.testing.assert_close(initial, torch.zeros_like(initial), rtol=0, atol=0)
  initial.sum().backward()
  assert model.net[-1].bias.grad.abs().min() > 0
  with torch.no_grad():
    model.net[-1].bias.fill_(2.0)
  correction = model(inputs, visible)
  assert correction[0].abs().min() > 0
  assert correction[0].abs().max() <= 0.12
  torch.testing.assert_close(correction[1], torch.zeros(29), rtol=0, atol=0)
