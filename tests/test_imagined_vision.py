from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from supermimic.models import MultiEncoderAE
from supermimic.tasks.imagined_vision import (
  LATENT_DIM,
  PIXELS,
  DepthVision,
  pack_depth,
  unpack_depth,
)


def test_depth_encoding_preserves_misses_and_metric_range():
  ranges = torch.tensor([float("nan"), float("inf"), -1.0, 0.0, 0.125, 1.25, 2.5, 2.6])
  pixels = pack_depth(ranges)
  assert pixels.dtype == torch.uint8
  assert pixels.tolist()[:4] == [0, 0, 0, 0]
  assert pixels[-1].item() == 0
  torch.testing.assert_close(unpack_depth(pixels)[4:7], ranges[4:7], atol=0.005, rtol=0)


@pytest.mark.parametrize("mode", ["render", "latent"])
def test_visual_objectives_and_stop_gradient(mode):
  model = DepthVision(mode, 7, 3)
  camera = torch.rand(2, 4, PIXELS) * 2.5
  state = torch.randn(2, 3, 7, requires_grad=True)
  action = torch.randn(2, 4, 3)
  recon, forecast = model.losses(pack_depth(camera), state, action)
  assert torch.isfinite(recon) and recon > 0
  if mode == "latent":
    assert torch.isfinite(forecast) and forecast > 0
    forecast.backward(retain_graph=True)
    assert model.transition[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
  else:
    assert forecast is None and model.transition is None
  recon.backward()
  assert (
    model.encoder[0].weight.grad.norm() > 0 and model.decoder[0].weight.grad.norm() > 0
  )

  # The real Actor.forward must retain the action/BPTT gradient but block visual gradients.
  z = model.encode(camera[:, 0])
  base_action = torch.randn(2, 3, requires_grad=True)
  actor = SimpleNamespace(
    decode_actions=lambda _obs: base_action,
    foot_depth_fusion=None,
    visual_action=nn.Linear(LATENT_DIM, 3, bias=False),
  )
  output = MultiEncoderAE.forward(
    actor, TensorDict({"camera_latent": z}, batch_size=[2])
  )
  g_action, g_visual = torch.autograd.grad(
    output.sum(), (base_action, z), allow_unused=True
  )
  assert g_action.count_nonzero() == 6 and g_visual is None


def test_initial_latent_forecast_is_stable_for_a_full_policy_rollout():
  model = DepthVision("latent", 7, 3)
  z = model.encode(torch.rand(2, PIXELS))
  predicted = z
  for _ in range(16):
    predicted = model.next_latent(predicted, torch.randn(2, 7) * 50, torch.randn(2, 3))
  # The first transition normalizes the encoded feature again. Its variance
  # after the encoder depends on the pre-LayerNorm activations and epsilon.
  expected = torch.nn.functional.layer_norm(z, (LATENT_DIM,))
  torch.testing.assert_close(predicted, expected, atol=2.0e-5, rtol=2.0e-5)


def test_flat_dual_head_transfer_preserves_actions_and_rejects_missing_base_weight(tmp_path):
  import json
  from dataclasses import asdict

  from supermimic.runners.supermimic_runner import (
    SuperMimicRunner,
    SuperMimicRunnerCfg,
    WorldModel,
  )

  cfg = SuperMimicRunnerCfg(residual_correction=True, residual_shared_trunk=True, shared_head_width=128,
                            critic_variant="sim-calibrated", sim_calibration_weight=0.0)
  (tmp_path / "run_manifest.json").write_text(json.dumps({"runner_cfg": asdict(cfg)}))
  ctx = nn.Linear(2, 2)
  flat_policy = nn.Sequential(nn.Linear(7, 3))
  flat_physics = nn.Sequential(nn.Linear(8, 4))
  flat_critic = nn.Sequential(nn.Linear(6, 1))
  base_actor = {"decoder." + k: v for k, v in flat_policy.state_dict().items()}
  torch.save(base_actor, tmp_path / "actor.pt")
  torch.save(flat_critic.state_dict(), tmp_path / "critic.pt")
  torch.save({"dynamics": {"mlp." + k: v for k, v in flat_physics.state_dict().items()},
              "sim_context": ctx.state_dict(), "shared_head_width": 128}, tmp_path / "world_model.pt")
  cfg.terrain_height, cfg.terrain_inputs, cfg.depth_fusion, cfg.imagined_vision = .09, "actor-world", "concat", "latent"
  cfg.init_model_dir = str(tmp_path)
  actor = nn.Module()
  actor.decoder = nn.Sequential(nn.Linear(7+15, 3))
  actor.visual_action = nn.Linear(LATENT_DIM, 3, bias=False)
  actor.foot_depth_fusion = nn.Linear(4, 2)
  dynamics = nn.Module()
  dynamics.mlp = nn.Sequential(nn.Linear(8+15, 4))
  dynamics.vision = nn.Linear(5, 2)
  dynamics.residual_mlp, dynamics.shared_head_width = None, 128
  critic = nn.Sequential(nn.Linear(6+15, 1))
  runner = SimpleNamespace(cfg=cfg, device="cpu", policy_learner=SimpleNamespace(actor=actor,critic=critic),
                           _dynamics=dynamics, _wm=WorldModel(None,dynamics,ctx,0,3))
  SuperMimicRunner._load_initial_model(runner)
  x, extra = torch.randn(8, 8), torch.randn(8, 15)
  torch.testing.assert_close(actor.decoder(torch.cat([x[:,:7],extra],1)), flat_policy(x[:,:7]))
  torch.testing.assert_close(dynamics.mlp(torch.cat([x,extra],1)), flat_physics(x))
  torch.testing.assert_close(critic(torch.cat([x[:,:6],extra],1)), flat_critic(x[:,:6]))
  assert actor.visual_action.weight.count_nonzero() == 0
  del base_actor["decoder.0.bias"]
  torch.save(base_actor, tmp_path / "actor.pt")
  with pytest.raises(RuntimeError, match="decoder.0.bias"):
    SuperMimicRunner._load_initial_model(runner)
