from types import SimpleNamespace

import pytest
import torch

from supermimic.tasks.foot_depth_actor import MeasuredPatch
from supermimic.tasks.rough_reference import RoughFootReference, rough_body_tracking


class Feet:
  def __init__(self):
    self.ahead = torch.tensor([4, 8, 12])
    self.body_indices = [1, 3]
    self.xyz = torch.zeros(13, 2, 3)
    self.xyz[:, 0, 0] = torch.arange(13) * 0.04
    self.xyz[:, 0, 2] = 0.1
    self.xyz[:, 1, 0] = -0.08
    self.motion = SimpleNamespace(
      motion_ends=torch.tensor([13]),
      root_pos_w=torch.zeros(13, 3),
      root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(13, 1),
    )

  def __call__(self, time):
    index = (time[:, None] + self.ahead).clamp_max(12)
    return torch.nn.functional.pad(self.xyz[index].transpose(1, 2), (0, 6))


@pytest.mark.parametrize("mode", ["oracle", "camera"])
def test_relative_height_and_support(mode):
  feet = Feet()
  root = torch.zeros(2, 7)
  root[:, 3] = 1
  time = torch.tensor([0, 12])
  patch = MeasuredPatch(torch.zeros(1, 2))
  origin = root[:, [0, 1, 3, 4, 5, 6]]
  for slope in (0.0, 0.1):

    def height(xy):
      return 0.25 + slope * xy[..., 0]

    z = height(patch.crop_xy)
    image = torch.stack([10 * z, torch.ones_like(z)]).expand(2, -1, -1, -1).clone()
    patch.bind(image, origin)
    ref = RoughFootReference(feet, mode, SimpleNamespace(height_at=height))
    future, now, after = ref.targets(time, root, patch)
    if mode == "camera":
      replay = MeasuredPatch(torch.zeros(1, 2))
      replay.bind_replay({
        "depth_image": image.flatten(1).half(),
        "depth_origin": origin,
      })
      for live, restored in zip((future, now, after), ref.targets(time, root, replay), strict=True):
        torch.testing.assert_close(live, restored, atol=0.0003, rtol=0)
    delta = future - feet(time)
    assert not delta[..., [0, 1, 3, 4, 5, 6, 7, 8]].any()
    assert not delta[:, 1].any()
    torch.testing.assert_close(
      delta[0, 0, :, 2], (slope * torch.tensor([0.24, 0.40, 0.56]) - 0.001).clamp_min(0)
    )
    torch.testing.assert_close(after[-1], now[-1])
    if slope == 0:
      assert torch.equal(future, feet(time))
    else:
      body = (torch.zeros(2, 4, 3), None, torch.zeros(2, 4, 3), None)
      pos, _, vel, _ = ref.body_targets(body, now, after, 0.02)
      cmd = SimpleNamespace(
        body_pos_relative_w=body[0],
        robot_body_pos_w=pos,
        body_lin_vel_w=body[2],
        robot_body_lin_vel_w=vel,
        rough_ankle_targets=(feet.body_indices, now, (after - now) / 0.02),
      )
      env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd))
      for velocity in (True, False):
        torch.testing.assert_close(
          rough_body_tracking(env, "motion", 0.3, velocity), torch.ones(2)
        )
    if mode == "camera":
      # Unknown support or unknown landing must not use the oracle.
      feet.xyz[:, 0, 0].neg_()
      image[:, 1] = patch.crop_xy[..., 0] >= -0.09
      patch.bind(image, origin)
      back, _, _ = ref.targets(time, root, patch)
      assert torch.equal(back, feet(time))
      feet.xyz[:, 0, 0].neg_()


@pytest.mark.parametrize("mode", ["off", "nominal", "camera", "oracle"])
def test_g1_training_reward_configuration(mode):
  import mjlab.tasks
  from mjlab.tasks.registry import load_env_cfg
  from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  original = {name: term.func for name, term in cfg.rewards.items()}
  runner = SuperMimicRunnerCfg(
    depth_fusion="concat", terrain_inputs="actor-world", terrain_height=0.09
  )
  assert runner.rough_reference == "off"
  runner.rough_reference = mode
  runner.configure_terrain(cfg)
  changed = {
    name for name, term in cfg.rewards.items() if original[name] is not term.func
  }
  assert changed == (
    {"motion_body_pos", "motion_body_lin_vel"}
    if mode in ("oracle", "camera")
    else set()
  )


def test_no_support_or_clip_boundary_leak():
  feet = Feet()
  feet.motion.motion_ends = torch.tensor([5, 13])
  # Make the test double's goal window respect the same two motion clips.
  time = torch.tensor([4, 12])
  root = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).expand(2, -1)

  class InvisibleSupport:
    def sample(self, root, xy):
      return torch.stack([xy[..., 0], (xy[..., 0] >= 0).float()], -1)

  future, now, after = RoughFootReference(feet, "camera", None).targets(
    time, root, InvisibleSupport()
  )
  assert torch.equal(future, feet(time))
  assert not now.any() and not after.any()
  ref = RoughFootReference(
    feet, "oracle", SimpleNamespace(height_at=lambda xy: xy[..., 0] * 0.1)
  )
  future, now, after = ref.targets(time, root, None)
  dz = future[..., 2] - feet(time)[..., 2]
  torch.testing.assert_close(dz, now.unsqueeze(-1).expand_as(dz))
  torch.testing.assert_close(after, now)


def test_support_switch_uses_continuous_position_and_velocity():
  feet = Feet()
  feet.xyz[:, :, 0] = torch.tensor([0.2, -0.2])
  feet.xyz[0, :, 2] = torch.tensor([0., 0.001])
  feet.xyz[1, :, 2] = torch.tensor([0.001, 0.])
  root = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]])
  ground = SimpleNamespace(height_at=lambda xy: xy[..., 0] * 0.125)
  reference = RoughFootReference(feet, "oracle", ground)
  _, now, after = reference.targets(torch.tensor([0]), root, None)
  _, next_now, _ = reference.targets(torch.tensor([1]), root, None)
  torch.testing.assert_close(after, next_now, atol=1e-7, rtol=0)
  assert (after - now).abs().max() < 0.003
  body = (torch.zeros(1, 4, 3), None, torch.zeros(1, 4, 3), None)
  _, _, velocity, _ = reference.body_targets(body, now, after, 0.02)
  torch.testing.assert_close(velocity[0, feet.body_indices, 2], ((next_now - now) / 0.02)[0])


def test_terrain_observe_drops_stale_hits_and_robot_returns():
  from supermimic.tasks.instinct_camera import TorsoCamera

  root = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).expand(2, -1)
  data = SimpleNamespace(
    output={"distance_to_image_plane": torch.ones(2, 27, 48, 1)},
    pos_w=root[:, :3], quat_w_world=root[:, 3:7],
  )
  rays = torch.tensor([1., 0., 0.]).expand(2, 1296, -1)
  hits = torch.zeros(2, 1296, dtype=torch.long)
  hits[1, 1], hits[1, 2] = 1, -1
  camera = SimpleNamespace(
    data=data, ray_directions=rays, ray_starts=torch.zeros_like(rays), num_rays=1296,
    static_geom=torch.tensor([True, False]), refresh_mask=torch.tensor([False, True]),
    _read_backend_hit_geom_ids=lambda: hits, noisy_image=torch.empty(2, 576),
    pipeline=SimpleNamespace(clean=lambda image: image),
  )
  masks = []
  memory = SimpleNamespace(update_cloud=lambda points, valid, pos, quat: masks.append(valid))
  TorsoCamera.observe(camera, memory, root, terrain_only=True)
  assert not masks[0][0].any()
  assert masks[0][1, 0] and not masks[0][1, 1:3].any()
