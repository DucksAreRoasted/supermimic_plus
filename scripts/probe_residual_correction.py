"""Check the optional residual dynamics branch on CUDA without a simulator."""
import torch

from supermimic.models import Dynamics
from supermimic.representations import create_representation


def main():
  device = "cuda:0"
  torch.manual_seed(42)
  rep = create_representation(
    "gencoord", 71, num_actions=29, dt=0.02, decimation=4,
    default_joint_pos=torch.zeros(29), actor_obs_mode="rep_state",
  )
  def build(enabled):
    return Dynamics(rep, 29, 2, torch.ones(29), torch.zeros(29),
                    (64, 64), residual_correction=enabled).to(device).eval()
  base, residual = build(False), build(True)
  residual.load_state_dict(base.state_dict(), strict=False)
  assert base.residual_mlp is None
  state = torch.randn(16, rep.rep_state_dim, device=device) * 0.1
  aux = torch.zeros(16, 7, device=device)
  aux[:, 3] = 1
  action = torch.randn(16, 29, device=device) * 0.1
  context = torch.zeros(16, 2, device=device)
  baseline = base(state, action, aux, context)
  residual.residual_scale = 0.0
  disabled = residual(state, action, aux, context)
  for a, b in zip(baseline, disabled):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
  residual.residual_scale = 0.1
  output = residual(state, action, aux, context)
  assert all(torch.isfinite(x).all() for x in output)
  assert any(not torch.equal(a, b) for a, b in zip(output, baseline))
  optimizer = torch.optim.Adam(residual.parameters(), lr=1e-4)
  before = residual.residual_mlp[-1].weight.detach().clone()
  sum(x.square().mean() for x in output).backward()
  for module in (residual.mlp, residual.residual_mlp):
    grads = [p.grad for p in module.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
  optimizer.step()
  assert not torch.equal(before, residual.residual_mlp[-1].weight)
  print("PASS: disabled equivalence, finite outputs, both branch gradients, residual update")


if __name__ == "__main__":
  main()
