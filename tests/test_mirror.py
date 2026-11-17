import torch

from supermimic.mirror import joint_mirror_permutation, joint_mirror_signs, mirror_full_state


def test_joint_name_mirror_is_involution():
  names = ["left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee", "torso"]
  perm, pairs = joint_mirror_permutation(names)
  assert perm == [1, 0, 3, 2, 4]
  assert len(pairs) == len(names)
  assert [perm[i] for i in perm] == list(range(len(names)))


def test_joint_name_mirror_axis_signs():
  names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "torso"]
  assert joint_mirror_signs(names) == [1.0, -1.0, -1.0, 1.0]


def test_full_state_mirror_is_involution():
  names = [f"joint_{i}" for i in range(29)]
  perm, _ = joint_mirror_permutation(names)
  perm = torch.tensor(perm)
  sign = torch.tensor(joint_mirror_signs(names))
  state = torch.randn(3, 71)
  state[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
  mirrored = mirror_full_state(state, perm, sign)
  restored = mirror_full_state(mirrored, perm, sign)
  torch.testing.assert_close(restored, state, atol=1e-5, rtol=1e-5)
