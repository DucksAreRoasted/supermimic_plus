from types import SimpleNamespace

import pytest
import torch

from supermimic.tasks.foot_depth_actor import FootDepthFusion, FutureFeet, MeasuredPatch


def joints():
  return ["waist_yaw_joint"] + [
    f"{side}_{joint}_joint"
    for joint in (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
    )
    for side in ("left", "right")
  ]


@pytest.mark.parametrize("mode", ["concat", "query"])
def test_fusion_starts_identical_and_modulates_each_leg_independently(mode):
  head = FootDepthFusion(mode, joints())
  action = torch.randn(3, len(joints()))
  depth = torch.randn(3, 49, 4)
  depth[..., 3] = 1
  poses = torch.randn(3, 2, 3, 9)
  torch.testing.assert_close(head(action, depth, poses), action, rtol=0, atol=0)
  torch.nn.init.normal_(head.modulator[-1].weight, std=0.1)
  out = head(action, depth, poses)
  left_moved = poses.clone()
  left_moved[:, 0] += 1
  changed = head(action, depth, left_moved)
  assert not torch.allclose(out[:, head.legs[0]], changed[:, head.legs[0]])
  assert torch.equal(out[:, head.legs[1]], changed[:, head.legs[1]])
  assert torch.equal(out[:, 0], action[:, 0])
  depth[..., 3] = 0
  unknown = head(action, depth, poses)
  depth[..., 2] = float("nan")
  assert torch.equal(head(action, depth, poses), unknown) and unknown.isfinite().all()


def test_query_routes_depth_by_left_and_right_future_positions():
  head = FootDepthFusion("query", joints())
  coords = MeasuredPatch(torch.zeros(1, 2)).token_xy
  depth = torch.cat([coords, torch.ones(49, 2)], -1)[None]
  poses = torch.zeros(1, 2, 3, 9)
  poses[:, :, :, 0] = 0.3
  poses[:, 0, :, 1], poses[:, 1, :, 1] = 0.30, -0.30
  with torch.no_grad():
    for net in (head.depth, head.pose):
      for p in net.parameters():
        p.zero_()
    head.depth[-1].bias.fill_(1)
  inputs = []
  hook = head.modulator.register_forward_pre_hook(
    lambda _, args: inputs.append(args[0].detach())
  )
  head(torch.zeros(1, 13), depth, poses)
  hook.remove()
  weights = inputs[0][0, :, : 49 * 12].reshape(2, 49, 12)[..., 0]
  torch.testing.assert_close(coords[weights.argmax(-1)], poses[0, :, 0, :2])


def test_measured_crop_tracks_pelvis_translation_and_rotation():
  offsets = torch.tensor([[0.16, 0.0], [0, 0.16]])
  patch = MeasuredPatch(offsets)
  x, y = patch.crop_xy.unbind(-1)
  image = torch.stack([x + 2 * y, torch.ones_like(x)])[None]
  root = torch.tensor([[0.08, 0.08, 0.9, 2**-0.5, 0, 0, 2**-0.5]])
  patch.bind(image, torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
  torch.testing.assert_close(
    patch.features(root), torch.tensor([[0.56, 0.08]]), atol=2e-6, rtol=0
  )
  root[:, :2] = 10
  assert not patch.tokens(root)[..., 3].any()


def test_reference_does_not_cross_clip_ends_and_uses_orientation():
  frames = torch.arange(8).float()
  xyz = torch.zeros(8, 2, 3)
  xyz[..., 0] = frames[:, None]
  quat = torch.zeros(8, 2, 4)
  quat[..., 0] = 1
  quat[:, 1] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
  motion = SimpleNamespace(
    joint_pos=frames,
    root_pos_w=torch.zeros(8, 3),
    root_quat_w=quat[:, 0],
    motion_ends=torch.tensor([4, 8]),
    body_state=lambda f: (xyz[f], quat[f]),
  )
  cmd = SimpleNamespace(
    motion=motion,
    cfg=SimpleNamespace(body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
  )
  p = FutureFeet(cmd)(torch.tensor([2, 5]))
  torch.testing.assert_close(
    p[:, :, :, 0], torch.tensor([3.0, 7.0])[:, None, None].expand(-1, 2, 3)
  )
  torch.testing.assert_close(p[0, 1, 0, 3:6], torch.tensor([0.0, 1.0, 0.0]))


def test_cached_reference_matches_body_and_feet_at_clip_boundaries():
  rng = torch.Generator().manual_seed(42)
  count = 1059
  xyz, lin_vel, ang_vel = (torch.randn(count, 3, 3, generator=rng) for _ in range(3))
  quat = torch.nn.functional.normalize(torch.randn(count, 3, 4, generator=rng), dim=-1)
  bodies = (xyz, quat, lin_vel, ang_vel)
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=["base", "right_ankle_roll_link", "left_ankle_roll_link"]),
    motion=SimpleNamespace(joint_pos=xyz[:, 0], root_pos_w=xyz[:, 0], root_quat_w=quat[:, 0],
                           motion_ends=torch.tensor([1, 7, 1015, count]),
                           body_state=lambda t: tuple(x[t] for x in bodies)),
  )
  reference = FutureFeet(command)
  cached = FutureFeet(command, cache_reference=True)
  times = torch.tensor([1058, 0, 6, 1, 6, 1014, 1015, 1023, 1024])
  torch.testing.assert_close(cached(times), reference(times), rtol=0, atol=0)
  for saved, expected in zip(cached.body_state(times), command.motion.body_state(times), strict=True):
    torch.testing.assert_close(saved, expected, rtol=0, atol=0)


def test_combined_terrain_sample_preserves_values_and_root_gradient():
  patch = MeasuredPatch(torch.tensor([[0.0, 0.0], [0.2, -0.2]]))
  root = torch.tensor([[0.2, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], requires_grad=True)
  image = torch.rand(1, 2, 25, 25)
  image[:, 1] = 1.0
  patch.bind(image, torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
  expected = patch.features(root), patch.tokens(root)
  actual = patch.features_and_tokens(root)
  for a, b in zip(actual, expected, strict=True):
    torch.testing.assert_close(a, b)
  def grad(parts):
    return torch.autograd.grad(sum(item.sum() for item in parts), root)[0]
  torch.testing.assert_close(grad(actual), grad(expected))
