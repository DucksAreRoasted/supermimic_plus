"""Check paired G1 reset poses against MuJoCo ground rays on the training course."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from train_g1_dual_head_depth import CLIPS, ROOT, create_env, load_env_cfg, scratch_config


def capture(enabled, count):
  runner = scratch_config(40000, 64, 42, False, 4096, terrain_height_reset=enabled)
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  runner.configure_terrain(cfg)
  cfg.seed = 42
  cfg.sim.mujoco.iterations = cfg.sim.mujoco.ls_iterations = 60
  cfg.events = {k: v for k, v in cfg.events.items() if k in ("foot_friction", "base_com", "encoder_bias")}
  cfg.commands["motion"].motion_file = [
    str(ROOT / "data/modelscope/LAFAN1_Retargeting_Dataset/npz" / (clip + ".npz")) for clip in CLIPS
  ]
  cfg.commands["motion"].sampling_mode = "external"
  env = create_env(cfg, count, "cuda:0")
  try:
    sim = env.unwrapped
    cmd = sim.command_manager.get_term("motion")

    def state():
      return tuple(t.detach().cpu().numpy().copy()
                   for t in (sim.sim.data.qpos, sim.sim.data.qvel, cmd.time_steps, cmd.anchor_pos_w))

    sim.reset(seed=1903)
    initial = state()
    ids = torch.arange(1, count, 2, device=sim.device)
    sim.reset(seed=1907, env_ids=ids)
    subset = state()
    for before, after in zip(initial, subset, strict=True):
      np.testing.assert_array_equal(before[::2], after[::2])

    for i in range(count):
      cmd.reset_to_frame(torch.tensor([i], device=sim.device), 100 + 37 * i)
    sim.sim.forward()
    by_frame = state()

    sim.episode_length_buf.fill_(sim.max_episode_length - 1)
    torch.manual_seed(1913)
    _, _, dones, _ = env.step(torch.zeros((count, env.num_actions), device=sim.device))
    assert dones.all() and (sim.episode_length_buf == 0).all()
    return cfg.scene.spec_fn, dict(reset=initial, subset=subset, frame=by_frame, step_timeout=state())
  finally:
    env.close()


def ground_rays(course):
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
  course(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  def heights(xy):
    hits = []
    for x, y in xy:
      distance = mujoco.mj_ray(model, data, [x, y, 2], [0, 0, -1],
                               None, True, -1, np.empty(1, np.int32))
      assert distance >= 0
      hits.append(2 - distance)
    return np.asarray(hits)

  return heights


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("out", type=Path)
  args = parser.parse_args()
  count = 32
  course, raw = capture(False, count)
  other_course, raised = capture(True, count)
  assert course == other_course
  ray_height = ground_rays(course)
  report = {}
  for stage, (base_q, base_v, base_frame, base_reference) in raw.items():
    q, v, frame, reference = raised[stage]
    np.testing.assert_array_equal(q[:, :2], base_q[:, :2])
    np.testing.assert_array_equal(q[:, 3:], base_q[:, 3:])
    np.testing.assert_array_equal(v[:, :3], base_v[:, :3])
    np.testing.assert_array_equal(v[:, 6:], base_v[:, 6:])
    # mjlab rotates world angular velocity into the root frame in float32.
    np.testing.assert_allclose(v[:, 3:6], base_v[:, 3:6], rtol=0, atol=2 * np.finfo(v.dtype).eps)
    np.testing.assert_array_equal(frame, base_frame)
    np.testing.assert_array_equal(reference, base_reference)
    expected = ray_height(q[:, :2])
    actual = q[:, 2] - base_q[:, 2]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    assert np.max(actual) > 0.01
    report[stage] = dict(envs=count, max_lift_cm=float(actual.max() * 100),
                         max_abs_error_m=float(np.abs(actual - expected).max()),
                         max_angular_velocity_delta_rad_s=float(np.abs(v[:, 3:6] - base_v[:, 3:6]).max()))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps(report))


if __name__ == "__main__":
  main()
