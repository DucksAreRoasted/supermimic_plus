from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_apply

from supermimic.representations import (
  _assemble_goal,
  motion_anchor_window,
  motion_goal_window,
)
from supermimic.tasks.depth_mimic import (
  DepthBalanceCorrection,
  DepthMemory,
  WalkingCameraPattern,
  steer_motion_window,
)


def test_pinhole_ranges_reconstruct_slope_after_motion_and_expire():
  camera = WalkingCameraPattern()
  memory = DepthMemory(2, "cpu", 0.02, camera)
  origin, direction = camera.generate_rays(None, "cpu")
  p = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]])
  q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.0, 0.0, 0.70710678]])
  o = p[:, None] + quat_apply(
    q[:, None].expand(-1, len(origin), -1), origin.expand(2, -1, -1)
  )
  d = quat_apply(q[:, None].expand(-1, len(origin), -1), direction.expand(2, -1, -1))
  # Independent ray / z=.03+.09x-.04y plane intersections.
  ranges = (0.03 + 0.09 * o[..., 0] - 0.04 * o[..., 1] - o[..., 2]) / (
    d[..., 2] - 0.09 * d[..., 0] + 0.04 * d[..., 1]
  )
  ranges = torch.where((ranges > 0) & (ranges <= 2.5), ranges, -1.0)
  memory.update(ranges, p, q)
  query = torch.tensor([[[0.65, 0.12]], [[0.65, 0.12]]])
  z, valid = memory.at(query)
  assert valid.all()
  torch.testing.assert_close(
    z[:, 0],
    torch.tensor([0.03 + 0.09 * 0.65 - 0.04 * 0.12, 0.03 - 0.09 * 0.12 - 0.04 * 0.65]),
    atol=0.007,
    rtol=0,
  )
  world = torch.tensor([[[0.65, 0.12]], [[-0.12, 0.65]]])
  assert memory.at_world(world, max_age=0)[1].all()
  # A point that just moved behind both virtual cameras is still in memory.
  moved = p + torch.tensor([[0.8, 0.0, 0.0], [0.0, 0.8, 0.0]])
  memory.update(torch.full_like(ranges, -1.0), moved, q)
  old, known = memory.at(query - torch.tensor([0.8, 0.0]))
  torch.testing.assert_close(old, z)
  assert known.all()
  assert not memory.at_world(world, max_age=0)[1].any()
  torch.testing.assert_close(memory.at_world(world)[0], old)
  memory.reset(torch.tensor([True, False]))
  _, known = memory.at(query - torch.tensor([0.8, 0.0]))
  assert known[:, 0].tolist() == [False, True]
  memory.tick += memory.max_age + 1
  assert not memory.at(query - torch.tensor([0.8, 0.0]))[1].any()


def test_balancing_has_no_activation_on_flat_or_missing_pixels():
  names = [
    s + "_ankle_" + j + "_joint" for s in ("left", "right") for j in ("pitch", "roll")
  ]
  balance = DepthBalanceCorrection(names, torch.ones(4))
  heights = torch.ones(2, 15) * 0.17
  mask = torch.ones_like(heights, dtype=torch.bool)
  mask[1] = False
  assert balance.terrain_gate(heights, mask).eq(0).all()
  heights[:, 4:8] += 0.03
  assert balance.terrain_gate(heights, mask).tolist() == [1.0, 0.0]


def test_reference_steering_preserves_joint_track_and_is_identity_without_depth():
  t = torch.arange(60, dtype=torch.float32)
  root_pos = torch.stack([t * 0.02, torch.zeros_like(t), torch.ones_like(t)], -1)
  q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(len(t), 1)
  motion = SimpleNamespace(
    root_pos_w=root_pos,
    root_quat_w=q,
    motion_ends=torch.tensor([len(t)]),
    joint_pos=torch.zeros(len(t), 29),
    joint_vel=torch.zeros(len(t), 29),
  )
  times = torch.tensor([3])
  cmd = SimpleNamespace(motion=motion, time_steps=times, anchor_pos_w=root_pos[times])
  root = torch.cat([root_pos[times], q[times]], -1)
  root[:, 1] -= 0.2
  positions, rotations = motion_anchor_window(motion, times)
  _, goal = _assemble_goal(
    motion_goal_window(motion, times), root[:, :3], root[:, 3:7], positions, rotations
  )
  flat = steer_motion_window(goal, root, cmd, torch.zeros(1), 0.5)
  hill = steer_motion_window(goal, root, cmd, torch.ones(1), 0.5)
  assert torch.equal(flat, goal)
  assert torch.equal(hill[:, :-60], goal[:, :-60])
  assert not torch.equal(hill[:, -60:], goal[:, -60:])


def test_contact_adapter_uses_only_measured_support_and_limits_joint_corrections():
  from supermimic.tasks.depth_mimic import ContactDepthAdapter

  names = [
    side + "_" + joint + "_joint"
    for side in ("left", "right")
    for joint in ("ankle_pitch", "ankle_roll", "hip_pitch", "knee")
  ]
  feet = torch.tensor([[0.1, 0.12, 0.05], [-0.1, -0.12, 0.11]]).repeat(8, 1, 1)
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(8, 2, 1)
  motion = SimpleNamespace(
    joint_pos=torch.zeros(8, len(names)),
    root_pos_w=torch.zeros(8, 3),
    root_quat_w=quat[:, 0],
    motion_ends=torch.tensor([8]),
    fk=SimpleNamespace(joint_names=names),
    body_state=lambda t: (feet[t], quat[t], None, None),
  )
  command = SimpleNamespace(
    motion=motion,
    time_steps=torch.tensor([1]),
    cfg=SimpleNamespace(body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
  )
  scale = torch.full((len(names),), 0.4)
  adapter = ContactDepthAdapter(command, scale)
  action = torch.zeros(1, len(names))

  def surface(xy):
    return 0.10 * xy[..., 0] + 0.08 * xy[..., 1] + 0.05, torch.ones_like(
      xy[..., 0], dtype=torch.bool
    )

  delta = adapter.apply(action, SimpleNamespace(at=surface), command, 1.5) * scale
  lp = names.index("left_ankle_pitch_joint")
  lr = names.index("left_ankle_roll_joint")
  assert delta[0, lp] < -0.10 and 0 < delta[0, lr] <= 0.12
  assert delta.abs().max() <= 0.21
  assert action.eq(0.0).all()
  for height, visible in ((0.03, True), (0.25, False)):

    def ground(xy, height=height, visible=visible):
      return torch.full_like(xy[..., 0], height), torch.full_like(
        xy[..., 0], visible, dtype=torch.bool
      )

    torch.testing.assert_close(
      adapter.apply(action, SimpleNamespace(at=ground), command, 1.5),
      action,
      rtol=0,
      atol=0,
    )


def test_stride_regulates_excursion_without_retiming_or_changing_upper_body():
  from supermimic.representations import goal_window_indices
  from supermimic.tasks.depth_mimic import regulate_stride_window

  names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "elbow",
  ]
  t = torch.arange(60, dtype=torch.float32)
  q = torch.sin(t[:, None] * 0.12 + torch.arange(len(names)))
  motion = SimpleNamespace(
    joint_pos=q,
    joint_vel=0.12 * torch.cos(t[:, None] * 0.12 + torch.arange(len(names))),
    motion_ends=torch.tensor([len(t)]),
    fk=SimpleNamespace(joint_names=names),
  )
  times = torch.tensor([2])
  cmd = SimpleNamespace(
    time_steps=times,
    motion=motion,
    anchor_pos_w=torch.tensor([[0.4, 0.0, 0.8]]),
    anchor_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_anchor_lin_vel_w=torch.tensor([[0.8, 0.0, 0.0]]),
  )
  goal = motion_goal_window(motion, times)
  root = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]])
  assert torch.equal(goal, regulate_stride_window(goal, root, cmd, torch.zeros(1), 1.2))
  corrected = regulate_stride_window(goal, root, cmd, torch.ones(1), 1.2)
  n = goal_window_indices(motion, times).shape[1]
  raw = goal.view(1, n, 2, len(names))
  new = corrected.view_as(raw)
  torch.testing.assert_close(new[:, :, :, 4], raw[:, :, :, 4])
  torch.testing.assert_close(new[:, :, 0, :4].mean(1), raw[:, :, 0, :4].mean(1))
  assert (new[:, :, 0, :4].std(1) > raw[:, :, 0, :4].std(1)).all()


def test_stride_room_is_per_foot_and_requires_visible_support():
  from supermimic.tasks.depth_mimic import ContactDepthAdapter

  fore = torch.tensor([[0.0, 0.16], [0.04, 0.0]])
  cross = torch.zeros_like(fore)
  known = torch.ones(2, 2, 5, dtype=torch.bool)
  known[1, 1, 3] = False
  room = ContactDepthAdapter.stride_room((fore, cross, cross, known, cross), 1.5)
  torch.testing.assert_close(room, torch.tensor([[1.0, 0.0], [0.75, 0.0]]))
