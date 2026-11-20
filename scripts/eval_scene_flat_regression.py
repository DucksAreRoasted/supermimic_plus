"""Replay the saved 40-motion flat protocol with a scene-conditioned tracking actor."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from eval_walk_terrain import checkpoint_config
from mjlab.tasks.registry import load_env_cfg

from supermimic.representations import (
  GenCoordRepresentation,
  extract_actor_goal,
  root_waypoints,
)
from supermimic.runners.supermimic_runner import build_actor
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.walk_terrain import TerrainObservation, WalkTerrain


@torch.inference_mode()
def evaluate(checkpoint, protocol, starts_file, device):
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
  clips = protocol["clips"]
  for clip in clips:
    with open(clip["path"], "rb") as source:
      if hashlib.file_digest(source, "sha256").hexdigest() != clip["sha256"]:
        raise ValueError(f"Motion checksum differs for {clip['path']}")
  cfg.scene.spec_fn = WalkTerrain(0.0)
  cfg.seed = protocol["seed"]
  cfg.events = {}
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1024)
  for group in cfg.observations.values():
    group.enable_corruption = False
  command = cfg.commands["motion"]
  command.motion_file = [c["path"] for c in clips]
  command.sampling_mode = "start"
  command.pose_range, command.velocity_range, command.joint_position_range = (
    {},
    {},
    (0.0, 0.0),
  )
  env = create_env(cfg, len(clips), device)
  try:
    sim = env.unwrapped
    sim.reset(seed=cfg.seed)
    motion = sim.command_manager.get_term("motion")
    if motion.motion.motion_names != [Path(c["path"]).stem for c in clips]:
      raise ValueError("Motion ordering differs from the saved protocol")
    frames = [c["frames"] for c in clips]
    np.testing.assert_array_equal(motion.motion.motion_ends.cpu(), np.cumsum(frames))
    starts = np.r_[0, np.cumsum(frames)[:-1]] + [c["local_start"] for c in clips]
    for i, start in enumerate(starts):
      motion.reset_to_frame(torch.tensor([i], device=device), int(start))
    sim.scene.write_data_to_sim()
    sim.sim.forward()
    sim.sim.sense()
    motion.update_relative_body_poses()
    obs = sim.observation_manager.compute(update_history=True)
    with np.load(starts_file) as saved:
      for key, states in [
        ("qpos", sim.sim.data.qpos),
        ("qvel", sim.sim.data.qvel),
        ("frames", motion.time_steps),
      ]:
        np.testing.assert_array_equal(states.cpu(), saved[key], err_msg=key)
      for key in ("full_state", "goal"):
        np.testing.assert_allclose(
          obs[key].cpu(), saved[key], rtol=0, atol=1e-6, err_msg=key
        )
        obs[key] = torch.as_tensor(saved[key], device=device)
    runner = checkpoint_config(checkpoint)
    representation = GenCoordRepresentation(29, motion.robot.data.default_joint_pos[0])
    world = SimpleNamespace(
      representation=representation, num_actions=29, num_goals=obs["goal"].shape[-1]
    )
    actor = build_actor(runner, world, device).eval()
    actor.load_state_dict(
      torch.load(checkpoint / "actor.pt", map_location=device, weights_only=True)
    )
    n = len(clips)
    active = torch.ones(n, dtype=torch.bool, device=device)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    completed = torch.zeros_like(active)
    metrics = ("error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel")
    totals = {
      k: torch.zeros(n, device=device, dtype=torch.float64)
      for k in (*metrics, "return")
    }
    for t in range(max(frames)):
      motion._update_metrics()
      for key in metrics:
        totals[key] += torch.where(active, motion.metrics[key], 0.0)
      rep = world.representation.represent(obs["full_state"])
      td = build_actor_obs_td({"motion": extract_actor_goal(obs["goal"])}, rep[:, None])
      if runner.terrain_inputs != "blind":
        td["terrain"] = rep.new_zeros(n, TerrainObservation.num_features)
      if runner.use_root_waypoints:
        td["root_waypoints"] = root_waypoints(obs["goal"])
      action = torch.where(active[:, None], actor(td), 0.0)
      obs, reward, terminated, truncated, _ = sim.step(action)
      ended = sim.termination_manager.get_term("motion_end")
      if (active & truncated & ~ended & ~terminated).any():
        raise RuntimeError("Unexpected timeout during a flat regression clip")
      totals["return"] += torch.where(active, reward, 0.0)
      counts += active.long()
      completed |= active & ended & ~terminated
      active &= ~(terminated | truncated)
      if t % 1000 == 0:
        print(f"Frame {t}, {active.sum().item()} remaining", flush=True)
      if not active.any():
        break
    if active.any():
      raise RuntimeError("At least one reference did not reach an episode boundary")
    rows = [
      dict(
        motion=motion.motion.motion_names[i],
        steps=int(counts[i]),
        completed=bool(completed[i]),
        reward_per_reference_frame=float(totals["return"][i] / frames[i]),
        **{k: float(totals[k][i] / counts[i]) for k in metrics},
      )
      for i in range(n)
    ]
    return dict(
      checkpoint=str(checkpoint),
      completed=int(completed.sum()),
      motions=n,
      means={
        k: float(np.mean([r[k] for r in rows]))
        for k in (*metrics, "reward_per_reference_frame")
      },
      per_motion=rows,
    )
  finally:
    env.close()


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  for arg in ("checkpoint", "protocol", "starts", "output"):
    p.add_argument(arg, type=Path)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  result = evaluate(
    a.checkpoint, json.loads(a.protocol.read_text()), a.starts, a.device
  )
  a.output.parent.mkdir(exist_ok=True, parents=True)
  a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(result["completed"], result["means"])
