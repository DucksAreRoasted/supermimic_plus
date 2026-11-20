import math

import mujoco
import numpy as np
import torch

from supermimic.models_push import ContactDynamics, PushPolicy
from supermimic.tasks.push_box import PushBox, PushCourse
from supermimic.tasks.walk_terrain import TerrainObservation


def test_free_box_rests_on_the_same_incline_used_in_imagination():
  course = PushCourse(height=0.0, grade=0.04, box=PushBox(floor_friction=0.28))
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(
    name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 1.0)
  )
  course(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  tilt = -math.atan(course.grade)
  start_x = 0.96
  data.qpos[:] = [
    start_x,
    1.2,
    course.grade * start_x
    + course.box.half_size[2] * math.hypot(1, course.grade)
    + 0.0005,
    math.cos(tilt / 2),
    0.0,
    math.sin(tilt / 2),
    0.0,
  ]
  mujoco.mj_forward(model, data)
  xy = np.array([[-1.0, 2.0], [1.5, -0.3]])
  ray_z = [
    1.0
    - mujoco.mj_ray(
      model, data, [x, y, 1.0], [0.0, 0.0, -1.0], None, True, -1, np.empty(1, np.int32)
    )
    for x, y in xy
  ]
  np.testing.assert_allclose(
    TerrainObservation(course).height_at(torch.tensor(xy)), ray_z
  )
  mujoco.mj_step(model, data, nstep=1500)
  assert abs(data.qpos[0] - start_x) < 0.002
  assert abs(data.qpos[1] - 1.2) < 1e-5


def test_unreachable_palms_cannot_drive_stationary_box_or_change_robot():
  torch.manual_seed(51)
  model = ContactDynamics()
  for net in (model.ground, model.reaction, model.push):
    torch.nn.init.normal_(net[-1].weight, std=0.3)
    torch.nn.init.normal_(net[-1].bias, std=0.2)
  box = torch.tensor(
    [[0.0, 0.0, 0.41, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
  )
  far = torch.full((1, 2, 3), 2.0)
  physics = torch.tensor([[2 / 3, 0.3]])
  torch.testing.assert_close(model.advance_box(box, far, far + 1.0, physics), box)
  actual = model.robot_delta(
    torch.zeros(1, 68), torch.zeros(1, 29), box, far, far + 1.0, physics
  )
  torch.testing.assert_close(actual, torch.zeros_like(actual))
  hands = torch.tensor([[[-0.37, 0.14, 0.75], [-0.37, -0.14, 0.75]]])
  velocity = torch.zeros_like(hands, requires_grad=True)
  prediction = model.advance_box(box, hands, hands + 0.02 * velocity, physics)
  prediction[:, 0].sum().backward()
  assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum() > 0
  torch.testing.assert_close(prediction[:, 3:7].norm(dim=-1), torch.ones(1))


def test_action_residual_warmstarts_exactly_and_stays_bounded():
  policy = PushPolicy()
  inputs = [
    torch.randn(shape) for shape in ((9, 68), (9, 7), (9, 13), (9, 2, 3), (9, 2))
  ]
  output = policy(*inputs)
  assert output.shape == (9, 29) and torch.count_nonzero(output) == 0
  output.sum().backward()
  assert policy.net[-1].weight.grad.abs().sum() > 0
  with torch.no_grad():
    policy.net[-1].weight.fill_(100.0)
  assert policy(*inputs).abs().amax() <= 0.65


def test_solid_ledge_surfaces_match_observed_heights_and_resting_box():
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 1.0)
  )
  course = PushCourse(
    height=0.04,
    box=PushBox(
      center=(0.96, 0.0, 0.431), half_size=(0.34, 0.27, 0.41), floor_friction=0.3
    ),
  )
  course(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  # At a convex mesh crease mj_ray can miss both triangles at their shared
  # edge. Check the surface interiors and both one-sided limits at each crease.
  samples = -0.4 + (np.arange(257) + 0.5) * 2.4 / 257
  seams = np.array([0.15, 0.50, 1.35, 1.70])
  xy = np.array([[x, 3.0] for x in np.r_[samples, seams - 0.0001, seams + 0.0001]])
  ray_z = [
    1.0
    - mujoco.mj_ray(
      model, data, [x, y, 1.0], [0.0, 0.0, -1.0], None, True, -1, np.empty(1, np.int32)
    )
    for x, y in xy
  ]
  np.testing.assert_allclose(
    TerrainObservation(course).height_at(torch.tensor(xy)), ray_z, atol=1.0e-6
  )
  mujoco.mj_step(model, data, nstep=1500)
  assert abs(data.qpos[0] - 0.96) < 0.002 and abs(data.qpos[1]) < 0.002


def test_stopping_reference_reproduces_nominal_window_then_holds_pose_at_goal():
  from types import SimpleNamespace

  from supermimic.representations import NUM_GOAL_FRAMES, extract_actor_goal
  from supermimic.tasks.push_guidance import stopping_reference

  motion = SimpleNamespace(
    joint_pos=torch.arange(220.0)[:, None],
    joint_vel=torch.ones(220, 1),
    root_pos_w=torch.zeros(220, 3),
    root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(220, 1),
  )
  root = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
  box = torch.zeros(2, 13)
  goal = torch.tensor([[0.7, 0.0], [0.01, 0.0]])
  wm, rate = stopping_reference(
    motion, torch.tensor([30.0, 51.0]), torch.tensor([60, 110]), root, box, goal
  )
  frames = extract_actor_goal(wm)[:, : 2 * NUM_GOAL_FRAMES].reshape(
    2, NUM_GOAL_FRAMES, 2
  )
  torch.testing.assert_close(rate, torch.tensor([1.0, 0.0]))
  torch.testing.assert_close(
    frames[0, :, 0], torch.arange(30.0, 80.0, 5.0).clamp(max=60)
  )
  torch.testing.assert_close(frames[1, :, 0], torch.full((10,), 51.0))
  torch.testing.assert_close(frames[1, :, 1], torch.zeros(10))
