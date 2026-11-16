"""Read-only probe for G1 q/v coordinates and finite-difference consistency."""

from __future__ import annotations

import argparse
import glob

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--motion-file", required=True)
  p.add_argument("--device", default="cpu")
  args = p.parse_args()
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.scene.num_envs = 1
  cfg.commands["motion"].motion_file = sorted(glob.glob(args.motion_file))
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  obs, _ = env.reset()
  state = obs["full_state"]
  action_dim = env.action_manager.total_action_dim
  action = torch.zeros((1, action_dim), device=args.device)
  result = env.step(action)
  next_obs = result[0]
  next_state = next_obs["full_state"]
  q, v, q_next = state[..., 7:36], state[..., 42:71], next_state[..., 7:36]
  v_next = next_state[..., 42:71]
  dt = float(env.unwrapped.step_dt)
  fd_err = (q_next - q - dt * v).abs()
  semi_err = (q_next - q - dt * v_next).abs()
  print(f"full_state_shape={tuple(state.shape)}")
  print("layout=root_pos[0:3],root_quat[3:7],joint_pos[7:36],root_lin_vel[36:39],root_ang_vel[39:42],joint_vel[42:71]")
  print(f"dt={dt:.8f}")
  print(f"q_finite={bool(torch.isfinite(q).all())} v_finite={bool(torch.isfinite(v).all())} q_next_finite={bool(torch.isfinite(q_next).all())}")
  print(f"finite_difference_max_abs_error={fd_err.max().item():.6e}")
  print(f"finite_difference_mean_abs_error={fd_err.mean().item():.6e}")
  print(f"semi_implicit_max_abs_error={semi_err.max().item():.6e}")
  print(f"semi_implicit_mean_abs_error={semi_err.mean().item():.6e}")
  if not torch.isfinite(fd_err).all():
    raise SystemExit("non-finite state probe")


if __name__ == "__main__":
  main()
