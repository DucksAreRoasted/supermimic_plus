import mujoco
import numpy as np
import pytest
import torch

from supermimic.tasks.walk_terrain import TerrainObservation, WalkTerrain


@pytest.mark.parametrize("cross", [0.0, 0.04])
def test_height_queries_match_vertical_mujoco_rays(cross):
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.1])
  course = WalkTerrain(phase_cells=13, lateral_height=cross)
  course(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  xy = np.random.default_rng(42).uniform([-17, -17], [17, 17], (200, 2))
  expected = []
  for x, y in xy:
    hit = mujoco.mj_ray(
      model, data, [x, y, 1.0], [0.0, 0.0, -1.0], None, True, -1, np.empty(1, np.int32)
    )
    expected.append(1.0 - hit)
  got = TerrainObservation(course).height_at(torch.tensor(xy, dtype=torch.float32))
  np.testing.assert_allclose(got, expected, atol=3e-6)


def test_query_moves_and_rotates_with_imagined_root():
  query = TerrainObservation(WalkTerrain())
  root = torch.tensor([[0.93, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], requires_grad=True)
  heights = query(root)
  heights.sum().backward()
  assert torch.isfinite(root.grad).all() and root.grad[0, 0].abs() > 0
  turned = root.detach().clone()
  turned[:, 3] = 0
  turned[:, 6] = 1
  assert not torch.equal(heights, query(turned))


def test_flat_checkpoint_columns_preserve_actor_and_world_model_but_can_learn():
  from supermimic.models import Dynamics, MultiEncoderAE
  from supermimic.representations import create_representation
  from supermimic.runners.supermimic_runner import extend_terrain_projection
  from tests.conftest import (
    ACTION_DIM,
    LATENT_DIM,
    OBS_DIM,
    make_actor_obs,
    make_actor_specs,
  )

  query = TerrainObservation(WalkTerrain())
  opts = dict(
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    task_specs=make_actor_specs(),
    slot_mix={"motion": 1.0},
    latent_dim=LATENT_DIM,
    hidden_dims=(32,),
    latent_bc_weight=0.0,
    history_length=1,
  )
  flat_actor = MultiEncoderAE(**opts)
  actor = MultiEncoderAE(**opts, terrain_obs_dim=query.num_features)
  state = flat_actor.state_dict()
  extend_terrain_projection(state, actor, "decoder.0.weight")
  actor.load_state_dict(state)
  obs = make_actor_obs(8)
  obs["terrain"] = torch.randn(8, query.num_features)
  torch.testing.assert_close(flat_actor(obs), actor(obs))
  actor(obs).square().sum().backward()
  assert actor.decoder[0].weight.grad[:, -query.num_features :].abs().sum() > 0

  rep = create_representation(
    "gencoord",
    13 + 2 * ACTION_DIM,
    ACTION_DIM,
    0.02,
    4,
    torch.zeros(ACTION_DIM),
    "rep_state",
  )
  opts = dict(
    representation=rep,
    num_actions=ACTION_DIM,
    sim_ctx_dim=5,
    action_scale=torch.ones(ACTION_DIM),
    action_offset=torch.zeros(ACTION_DIM),
    hidden_dims=(32,),
    residual_correction=True,
    residual_shared_trunk=True,
  )
  flat_wm, wm = Dynamics(**opts), Dynamics(**opts, terrain_query=query)
  state = flat_wm.state_dict()
  extend_terrain_projection(state, wm, "mlp.0.weight")
  wm.load_state_dict(state)
  root = torch.tensor([[0.93, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]]).expand(8, -1)
  inputs = (
    torch.randn(8, rep.rep_state_dim),
    torch.randn(8, ACTION_DIM),
    root,
    torch.zeros(8, 5),
  )
  for a, b in zip(flat_wm(*inputs), wm(*inputs), strict=True):
    torch.testing.assert_close(a, b)
  wm(*inputs)[1].square().sum().backward()
  assert wm.mlp[0].weight.grad[:, -query.num_features :].abs().sum() > 0


def test_waypoints_use_xy_in_current_robot_frame_and_15_frame_lookahead():
  from supermimic.representations import NUM_GOAL_FRAMES, _assemble_goal, root_waypoints

  root = torch.tensor([[1.0, 2.0, 0.8]])
  quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # 180-degree yaw
  targets = root[:, None].expand(1, NUM_GOAL_FRAMES, 3).clone()
  targets[:, 0, 0] += 0.1
  targets[:, 3, 1] -= 0.2
  full, _ = _assemble_goal(
    torch.zeros(1, 2), root, quat, targets, quat[:, None].expand(1, NUM_GOAL_FRAMES, 4)
  )
  torch.testing.assert_close(
    root_waypoints(full), torch.tensor([[-0.2, 0.0, 0.0, 0.4]]).tanh()
  )
