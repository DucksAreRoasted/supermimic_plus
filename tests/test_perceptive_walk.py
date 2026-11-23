from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics

from supermimic import representations
from supermimic.representations import (
  _assemble_goal,
  goal_window_indices,
  motion_goal_window,
)
from supermimic.tasks.perceptive_walk import (
  LocalGroundScan,
  TerrainMotionWindow,
  walking_scan,
)


def ground_scan(surface, visible=None):
  points, _ = walking_scan().pattern.generate_rays(None, "cpu")
  x, y, _ = points.unbind(-1)
  points[:, 2] = surface(x, y)
  valid = torch.ones_like(x, dtype=torch.bool) if visible is None else visible(x, y)
  return LocalGroundScan(
    SimpleNamespace(
      distances=torch.where(valid, 1.0, -1.0)[None], hit_pos_w=points[None]
    )
  )


def test_scan_axes_interpolation_and_unknown_pixels():
  scan = ground_scan(lambda x, y: 0.2 * x + 0.1 * y, lambda x, y: x <= 0.5)
  points = torch.tensor([[[0.123, -0.263], [-0.612, 0.452], [0.9, 0.1], [0.0, 2.0]]])
  heights, valid = scan.at(points)
  torch.testing.assert_close(
    heights[:, :2], (points[..., 0] * 0.2 + points[..., 1] * 0.1)[:, :2]
  )
  assert valid.tolist() == [[True, True, False, False]]
  assert heights[~valid].eq(0).all()


@pytest.fixture
def motion_window(tmp_path):
  model = get_spec().compile()
  ankles = ["left_ankle_roll_link", "right_ankle_roll_link"]
  fk = ForwardKinematics(
    model,
    torch.arange(1, model.nbody),
    torch.tensor(model.jnt_qposadr[1:]),
    torch.tensor([model.body(name).id - 1 for name in ankles]),
  )
  # Identical standing poses with different reference origins and headings.
  for i, yaw in enumerate((0.0, np.pi / 2)):
    frames = 12
    q = np.zeros((frames, len(fk.joint_names)), np.float32)
    pos = np.tile([i * 3.0, 0.0, 0.8], (frames, 1))
    quat = np.tile([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], (frames, 1))
    np.savez(
      tmp_path / f"{i}.npz",
      joint_names=fk.joint_names,
      joint_pos=q,
      joint_vel=q,
      root_pos_w=pos,
      root_quat_w=quat,
      root_lin_vel_w=0 * pos,
      root_ang_vel_w=0 * pos,
    )
  motion = MotionLoader(str(tmp_path / "*.npz"), fk)
  robot = SimpleNamespace(
    data=SimpleNamespace(
      soft_joint_pos_limits=torch.tensor(model.jnt_range[1:], dtype=torch.float)[None]
    )
  )
  return TerrainMotionWindow(
    SimpleNamespace(motion=motion, robot=robot, cfg=SimpleNamespace(body_names=ankles)),
    0.02,
  )


def test_wm_goal_reuses_one_frame_lookup_and_preserves_root_gradients(motion_window, monkeypatch):
  motion = motion_window.motion
  times = torch.cat((torch.zeros_like(motion.motion_ends[:1]), motion.motion_ends - 1))
  root = torch.cat((motion.root_pos_w[times], motion.root_quat_w[times]), -1).detach().requires_grad_()
  idx = goal_window_indices(motion, times)
  expected = _assemble_goal(motion_goal_window(motion, times), root[:, :3], root[:, 3:7],
                            motion.root_pos_w[idx], motion.root_quat_w[idx])
  calls = []
  def counted(loader, frames):
    calls.append(frames)
    return goal_window_indices(loader, frames)
  monkeypatch.setattr(representations, "goal_window_indices", counted)
  rep = representations.GenCoordRepresentation(motion.joint_pos.shape[-1], motion.joint_pos[0])
  actual = rep.compute_wm_goal(motion, times, root)
  assert len(calls) == 1
  for a, b in zip(actual, expected, strict=True):
    torch.testing.assert_close(a, b)
  def gradient(goals):
    return torch.autograd.grad(sum(t.sum() for t in goals), root)[0]
  torch.testing.assert_close(gradient(actual), gradient(expected))


def test_flat_or_unknown_keeps_raw_goal(motion_window):
  window = motion_window
  motion = window.motion
  times = torch.tensor([11])
  idx = goal_window_indices(motion, times)
  root = torch.cat([motion.root_pos_w[times], motion.root_quat_w[times]], -1)
  _, expected = _assemble_goal(
    motion_goal_window(motion, times),
    root[:, :3],
    root[:, 3:],
    motion.root_pos_w[idx],
    motion.root_quat_w[idx],
  )
  for available in (True, False):
    scan = ground_scan(
      lambda x, y: 0 * x,
      lambda x, y, available=available: torch.full_like(x, available, dtype=torch.bool),
    )
    goal, bend, coverage = window.actor_goal(scan, times, expected)
    torch.testing.assert_close(goal, expected, rtol=0, atol=0)
    assert bend.eq(0).all()
    assert coverage.eq(float(available)).all()


def test_side_ridge_follows_heading_and_only_changes_raised_leg(motion_window):
  window = motion_window
  motion = window.motion
  scan = ground_scan(lambda x, y: (y > 0).float() * 0.05)
  joints = len(motion.fk.joint_names)
  for time in (
    11,
    12,
  ):  # adjacent clips with different heading, never difference across the seam
    times = torch.tensor([time])
    expected = motion_goal_window(motion, times)
    target, bend, coverage = window.actor_goal(scan, times, expected)
    n = bend.shape[1]
    q, v = target[:, : 2 * joints * n].reshape(1, n, 2, joints).unbind(2)
    assert q[..., window.cols[0, 1]].min() > 0.1
    assert q[..., window.cols[0, 0]].max() < 0
    keep = torch.ones(joints, dtype=torch.bool)
    keep[window.cols[0]] = False
    assert q[..., keep].eq(0).all()
    torch.testing.assert_close(v, torch.zeros_like(v), atol=1.0e-4, rtol=0)
    assert coverage.eq(1).all()
