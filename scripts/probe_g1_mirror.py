"""Read-only G1 mirror probe for joint order and roll/yaw signs."""

from __future__ import annotations

import glob

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.mirror import joint_mirror_permutation, joint_mirror_signs


def main() -> None:
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.scene.num_envs = 1
  cfg.commands["motion"].motion_file = glob.glob(
    "data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk1_subject1.npz"
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  model = env.unwrapped.sim.mj_model
  names = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.joint(i).type != 0
  ]
  assert len(names) == env.action_manager.total_action_dim == 29
  perm, pairs = joint_mirror_permutation(names)
  signs = joint_mirror_signs(names)
  assert [perm[i] for i in perm] == list(range(len(names)))
  assert all(signs[i] == signs[perm[i]] for i in range(len(names)))
  for name, sign in zip(names, signs, strict=True):
    token = name.lower()
    if "roll" in token or "yaw" in token:
      assert sign == -1.0, (name, sign)
    elif "pitch" in token or "knee" in token or "elbow" in token:
      assert sign == 1.0, (name, sign)
  x = torch.arange(29, dtype=torch.float32)
  mirrored = x[perm] * torch.tensor(signs)
  restored = mirrored[perm] * torch.tensor(signs)
  torch.testing.assert_close(restored, x)
  print("PASS: G1 29-joint permutation is involutive")
  for pair, sign in zip(pairs, signs, strict=True):
    print(f"  {pair}: sign={sign:+.0f}")


if __name__ == "__main__":
  main()
