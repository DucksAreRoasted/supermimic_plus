"""Joint-name driven mirror permutations for humanoid augmentation."""

from __future__ import annotations

import re

_SIDE = re.compile(r"^(.*[/])?(left|right|l|r)([_-].*)$", re.IGNORECASE)


def joint_mirror_permutation(joint_names: list[str]) -> tuple[list[int], list[str]]:
  """Return a permutation and signed swap labels using joint names only."""
  lookup = {name.lower(): i for i, name in enumerate(joint_names)}
  perm: list[int] = []
  pairs: list[str] = []
  for name in joint_names:
    match = _SIDE.match(name)
    if match is None:
      perm.append(lookup[name.lower()])
      pairs.append(name)
      continue
    prefix, side, suffix = match.groups()
    prefix = prefix or ""
    opposite = prefix + ("right" if side.lower() in {"left", "l"} else "left") + suffix
    short = prefix + ("r" if side.lower() in {"left", "l"} else "l") + suffix
    idx = lookup.get(opposite.lower(), lookup.get(short.lower()))
    if idx is None:
      raise ValueError(f"No mirror partner found for joint {name!r}")
    perm.append(idx)
    pairs.append(f"{name}<->{joint_names[idx]}")
  return perm, pairs


def joint_mirror_signs(joint_names: list[str]) -> list[float]:
  """Return the reflection sign for each joint's scalar coordinate.

  Reflection across the sagittal plane flips axial x/z rotations (roll/yaw)
  and preserves the y-axis pitch coordinate.  The returned signs are ordered
  like ``joint_names`` and are applied after the left/right permutation.
  """
  signs: list[float] = []
  for name in joint_names:
    token = name.lower().split("/")[-1]
    signs.append(-1.0 if ("roll" in token or "yaw" in token) else 1.0)
  return signs
