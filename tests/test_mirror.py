from supermimic.mirror import joint_mirror_permutation, joint_mirror_signs


def test_joint_name_mirror_is_involution():
  names = ["left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee", "torso"]
  perm, pairs = joint_mirror_permutation(names)
  assert perm == [1, 0, 3, 2, 4]
  assert len(pairs) == len(names)
  assert [perm[i] for i in perm] == list(range(len(names)))


def test_joint_name_mirror_axis_signs():
  names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "torso"]
  assert joint_mirror_signs(names) == [1.0, -1.0, -1.0, 1.0]
