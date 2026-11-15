"""Closed-loop G1 reference-to-action mapping probe."""

from __future__ import annotations

import argparse
import glob

import torch
import mjlab.tasks  # noqa: F401
import supermimic.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from supermimic.mirror import joint_mirror_permutation


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--motion-file", required=True)
  p.add_argument("--device", default="cpu")
  args = p.parse_args()
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.scene.num_envs = 1
  cfg.commands["motion"].motion_file = sorted(glob.glob(args.motion_file))
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.reset()
  mj_model = env.unwrapped.sim.mj_model
  names = [mj_model.joint(i).name for i in range(mj_model.njnt) if mj_model.joint(i).type != 0]
  perm, pairs = joint_mirror_permutation(names)
  term = env.unwrapped.action_manager.get_term(env.unwrapped.action_manager.active_terms[0])
  scale = term.scale.to(args.device).float().reshape(-1)
  offset = term._offset.to(args.device).float().reshape(-1)
  motion = env.unwrapped.command_manager.get_term("motion").motion
  q_ref = motion.joint_pos[0, : len(names)].to(args.device).float()
  action = (q_ref - offset) / scale
  q_recon = action * scale + offset
  mirrored = action[perm] * scale[perm] + offset[perm]
  q_expected = q_ref[perm]
  print(f"joints={len(names)}")
  print(f"mapping_max_abs_error={(q_recon-q_ref).abs().max().item():.6e}")
  print(f"mirror_target_max_abs_error={(mirrored-q_expected).abs().max().item():.6e}")
  print("mirror_pairs=" + ",".join(pairs))
  if (q_recon-q_ref).abs().max() > 1e-5 or (mirrored-q_expected).abs().max() > 1e-5:
    raise SystemExit("G1 action mapping probe failed")


if __name__ == "__main__":
  main()
