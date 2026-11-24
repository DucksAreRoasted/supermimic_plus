import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.forward_kinematics import ForwardKinematics

from supermimic.utils.warp_fk import WarpForwardKinematics


@pytest.mark.parametrize(
  "device",
  [
    "cpu",
    pytest.param(
      "cuda",
      marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable"
      ),
    ),
  ],
)
def test_fused_g1_tracks_all_bodies_and_input_gradients(device):
  model = get_spec().compile()
  tracked = unitree_g1_flat_tracking_env_cfg().commands["motion"].body_names
  rows = torch.tensor([model.body(name).id - 1 for name in tracked])
  original = ForwardKinematics(
    model,
    torch.arange(1, model.nbody),
    torch.tensor(model.jnt_qposadr[1:]),
    rows,
    device=device,
  )
  fused = WarpForwardKinematics(original)
  generator = torch.Generator(device=device).manual_seed(42)
  nj = len(original.joint_names)
  inputs = tuple(
    torch.randn(7, width, device=device, generator=generator).requires_grad_()
    for width in (3, 4, nj, 3, 3, nj)
  )
  expected = original(*inputs)
  actual = fused(*inputs)
  for got, want in zip(actual, expected, strict=True):
    torch.testing.assert_close(got, want, atol=2e-5, rtol=3e-5)
  weights = tuple(
    torch.randn(x.shape, generator=generator, device=device) for x in expected
  )
  desired_grad = torch.autograd.grad(expected, inputs, weights)
  actual_grad = torch.autograd.grad(actual, inputs, weights)
  for got, want in zip(actual_grad, desired_grad, strict=True):
    torch.testing.assert_close(got, want, atol=5e-5, rtol=6e-5)
