"""Numerical parity of G1 measured/imagined depth against independent CPU MuJoCo rays."""

import argparse
import json
from pathlib import Path

import mjlab.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat

import supermimic.tasks  # noqa: F401
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.instinct_camera import (
  OFFSET,
  ROTATION,
  TorsoRenderer,
  camera_config,
)


def fixtures(spec):
  for name, y, h in (("probe_wall", 3.0, 2.0), ("probe_step", 6.0, 0.40)):
    spec.worldbody.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(1.1, y, h / 2),
      size=(0.2, 1.0, h / 2),
      group=0,
    )


def ray_reference(model, qpos, cam, row):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  body = cam._require_attachment_frame_metadata().body_id
  R = data.xmat[body].reshape(3, 3)
  p = data.xpos[body] + R @ np.asarray(OFFSET)
  off = np.empty(9)
  mujoco.mju_quat2Mat(off, np.array(ROTATION) / np.linalg.norm(ROTATION))
  R = R @ off.reshape(3, 3)
  dirs = cam.ray_directions[row].cpu().numpy()
  origins = p + cam.ray_starts[row].cpu().numpy() @ R.T
  world_dirs = dirs @ R.T
  out, points, hits = [], [], []
  for o, v, d in zip(origins, world_dirs, dirs, strict=True):
    tag = np.array([-1], dtype=np.int32)
    walked = 0.0
    for _ in range(25):
      dist = mujoco.mj_ray(
        model,
        data,
        o + walked * v,
        v,
        np.array([1, 0, 1, 0, 0, 0], np.uint8),
        1,
        -1,
        tag,
      )
      if dist < 0 or walked + dist > 0.05:
        break
      walked += dist + 1e-4
    plane = (walked + dist) * d[0] if dist >= 0 else 1e6
    out.append(plane)
    points.append(o + (walked + dist) * v)
    hits.append(tag[0])
  return np.array(out), np.array(points), np.array(hits), p


@torch.no_grad()
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
  cfg.events = {}
  cfg.scene.spec_fn = fixtures
  cfg.scene.env_spacing = 0.0
  cfg.scene.sensors = (*cfg.scene.sensors, camera_config())
  cfg.commands["motion"].motion_file = str(args.motion.resolve())
  env = create_env(cfg, 4, "cuda:0")
  try:
    sim = env.unwrapped
    sim.reset(seed=42)
    robot = sim.scene["robot"]
    cam = sim.scene["walk_depth"]
    model = sim.sim.mj_model
    renderer = TorsoRenderer(sim, 4, "cuda:0")
    ids = torch.arange(4, device="cuda")
    root = robot.data.root_link_pose_w.clone()
    root[:, :] = root.new_tensor([0, 0, 0.80, 1, 0, 0, 0])
    root[:, 1] = root.new_tensor([0, 3, 6, 6])
    joint = robot.data.default_joint_pos.clone().expand(4, -1).contiguous()
    joint[3, robot.joint_names.index("waist_yaw_joint")] += 0.5
    joint[3, robot.joint_names.index("waist_pitch_joint")] += 0.3
    robot.write_root_link_pose_to_sim(root)
    robot.write_joint_position_to_sim(joint)
    sim.scene.write_data_to_sim()
    sim.sim.forward()
    cam.reset()
    sim.sim.sense()
    measured = cam.data.output["distance_to_image_plane"]
    raw = renderer.raw(root, joint, ids)
    bounded = (raw.clamp(0, 2) - measured.clamp(0, 2)).abs()
    report = dict(
      render_max_clipped_m=bounded.max().item(),
      render_clean_max=(cam.pipeline.clean(raw) - cam.pipeline.clean(measured))
      .abs()
      .max()
      .item(),
      raw_size=list(measured.shape[1:3]),
      processed_size=list(cam.pipeline.clean(raw).shape[-2:]),
      step_dt=sim.step_dt,
      period=cam.cfg.update_period,
      scenes={},
    )
    for i, name in enumerate(("floor", "wall", "step", "bent_turn")):
      ref, points, hits, p = ray_reference(
        model, sim.sim.data.qpos[i].cpu().numpy(), cam, i
      )
      depth = measured[i].flatten().cpu().numpy()
      err = abs(np.clip(ref, 0, 2) - np.clip(depth, 0, 2))
      corners = [0, 47, 26 * 48, 27 * 48 - 1, 13 * 48 + 24]
      valid = (ref > 0) & (ref <= 2) & (depth > 0) & (depth <= 2)
      d = cam.ray_directions[i]
      local = cam.ray_starts[i] + d * (measured[i].flatten() / d[:, 0])[:, None]
      pts = (
        (cam.data.pos_w[i] + local @ matrix_from_quat(cam.data.quat_w_world[i]).T)
        .cpu()
        .numpy()
      )
      xyz = np.linalg.norm(pts[valid] - points[valid], axis=-1)
      pose_error = float(np.linalg.norm(p - cam.data.pos_w[i].cpu().numpy()))
      panel = name in ("step", "wall")
      geom_hits = int((hits == model.geom("probe_" + name).id).sum()) if panel else None
      report["scenes"][name] = dict(
        p99_m=float(np.quantile(err, 0.99)),
        max_m=float(err.max()),
        corner_center_max_m=float(err[corners].max()),
        hit_p99_m=float(np.quantile(xyz, 0.99)),
        mount_error_m=pose_error,
        fixture_pixels=geom_hits,
      )
      assert (
        np.quantile(err, 0.99) < 0.003
        and np.quantile(xyz, 0.99) < 0.005
        and pose_error < 0.001
      )
      assert not panel or geom_hits > 10
    assert report["render_clean_max"] < 0.02
    # Verify the cached image is held inside one sensor period.
    before = cam.frame.clone()
    cam.update(0.004)
    sim.sim.sense()
    assert torch.equal(cam.frame, before)
    cam.update(1 / 60)
    sim.sim.sense()
    assert (cam.frame == before + 1).all()
    report["cadence_verified"] = True
    imgs = raw.repeat(48, 1, 1, 1)
    clean = cam.pipeline.clean(imgs)
    noisy = cam.pipeline.noised(imgs)
    assert (
      noisy.isfinite().all()
      and noisy.min() >= 0
      and noisy.max() <= 1
      and not torch.equal(noisy, clean)
    )
    report["noise"] = dict(
      clean_mean=clean.mean().item(),
      noisy_mean=noisy.mean().item(),
      noisy_std=noisy.std().item(),
      mean_abs_change=(noisy - clean).abs().mean().item(),
      zero_fraction=(noisy == 0).float().mean().item(),
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
  finally:
    env.close()


if __name__ == "__main__":
  main()
