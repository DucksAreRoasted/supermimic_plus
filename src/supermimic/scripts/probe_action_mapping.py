"""Offline checks for action-space and joint-name mirror mappings."""

from __future__ import annotations

import argparse

import torch

from supermimic.mirror import joint_mirror_permutation


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--joints", nargs="+", required=True)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  perm, pairs = joint_mirror_permutation(args.joints)
  g = torch.Generator().manual_seed(args.seed)
  scale = torch.rand(len(perm), generator=g) + 0.5
  offset = torch.randn(len(perm), generator=g) * 0.1
  q_ref = torch.randn(len(perm), generator=g) * 0.2 + offset
  action = (q_ref - offset) / scale
  reconstructed = action * scale + offset
  mirrored_action = action[perm]
  mirrored_q = mirrored_action * scale[perm] + offset[perm]
  expected = q_ref[perm]
  print(f"joints={len(perm)}")
  print(f"mapping_max_abs_error={(reconstructed - q_ref).abs().max().item():.3e}")
  print(f"mirror_max_abs_error={(mirrored_q - expected).abs().max().item():.3e}")
  print("mirror_pairs=" + ",".join(pairs))
  if not torch.allclose(reconstructed, q_ref, atol=1e-6):
    raise SystemExit("action mapping probe failed")
  if not torch.allclose(mirrored_q, expected, atol=1e-6):
    raise SystemExit("mirror mapping probe failed")


if __name__ == "__main__":
  main()
