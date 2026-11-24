"""Replay all LAFAN40 clips on flat ground or a paired irregular terrain."""

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
from supermimic.tasks.depth_mimic import DepthMemory, walking_camera
from supermimic.tasks.foot_depth_actor import FutureFeet, MeasuredPatch
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.walk_terrain import (
  IrregularWalkTerrain,
  TerrainObservation,
  WalkTerrain,
)


@torch.inference_mode()
def evaluate(checkpoint, protocol, starts_file, device, *, rough_seed=None, save_starts=False):
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
  runner = checkpoint_config(checkpoint)
  camera_cfg = walking_camera() if runner.depth_fusion != "none" else None
  if camera_cfg is not None or rough_seed is not None:
    cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg or walking_camera())
  clips = protocol["clips"]
  for clip in clips:
    with open(clip["path"], "rb") as source:
      if hashlib.file_digest(source, "sha256").hexdigest() != clip["sha256"]:
        raise ValueError(f"Motion checksum differs for {clip['path']}")
  cfg.scene.spec_fn = WalkTerrain(0.0)
  if rough_seed is not None:
    from supermimic.tasks.terrain_reset import TerrainResetMotionCommandCfg

    cfg.scene.spec_fn = IrregularWalkTerrain(height=0.09, lateral_height=0.04, landscape_seed=rough_seed)
    cfg.scene.env_spacing = 0.0
    cfg.commands["motion"] = TerrainResetMotionCommandCfg.from_motion(cfg.commands["motion"])
    cfg.sim.mujoco.iterations = cfg.sim.mujoco.ls_iterations = 60
  if runner.flat_depth_arm is not None:
    cfg.scene.env_spacing = 0.0
    cfg.sim.mujoco.iterations = cfg.sim.mujoco.ls_iterations = 60
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
    if save_starts:
      starts_file.parent.mkdir(parents=True, exist_ok=True)
      with starts_file.open("xb") as out:
        np.savez(out, qpos=sim.sim.data.qpos.cpu().numpy(), qvel=sim.sim.data.qvel.cpu().numpy(),
                 frames=motion.time_steps.cpu().numpy(), full_state=obs["full_state"].cpu().numpy(),
                 goal=obs["goal"].cpu().numpy(), rough_seed=-1 if rough_seed is None else rough_seed)
    with np.load(starts_file) as saved:
      if (rough_seed is not None or "rough_seed" in saved) and int(saved["rough_seed"]) != (-1 if rough_seed is None else rough_seed):
        raise ValueError("Saved starts belong to a different terrain")
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
    representation = GenCoordRepresentation(29, motion.robot.data.default_joint_pos[0])
    # The physical Actor only needs the reference FK joint order for foot fusion.
    representation.actor_obs_mode = runner.actor_obs
    representation.fk = motion.motion.fk
    representation.pelvis_motion_goal = runner.pelvis_motion_goal
    world = SimpleNamespace(
      representation=representation, num_actions=29, num_goals=obs["goal"].shape[-1]
    )
    actor = build_actor(runner, world, device).eval()
    actor.load_state_dict(
      torch.load(checkpoint / "actor.pt", map_location=device, weights_only=True)
    )
    vision = None
    if runner.imagined_vision != "none":
      from supermimic.tasks.imagined_vision import DepthVision

      vision = DepthVision(runner.imagined_vision, representation.rep_state_dim, 29).to(device).eval()
      bundle = torch.load(checkpoint / "world_model.pt", map_location=device, weights_only=True)
      vision.load_state_dict({k.removeprefix("vision."): v for k, v in bundle["dynamics"].items()
                              if k.startswith("vision.")})
      del bundle
    n = len(clips)
    ground = TerrainObservation(cfg.scene.spec_fn).to(device)
    if camera_cfg is not None:
      camera = sim.scene["walk_depth"]
      memory = DepthMemory(n, device, sim.step_dt, camera_cfg.pattern)
      patch = MeasuredPatch(ground.offsets)
      future_feet = FutureFeet(motion)
    active = torch.ones(n, dtype=torch.bool, device=device)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    completed = torch.zeros_like(active)
    reasons = {name: torch.zeros_like(active) for name in sim.termination_manager.active_terms}
    metrics = ("error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel")
    totals = {
      k: torch.zeros(n, device=device, dtype=torch.float64)
      for k in (*metrics, "return")
    }
    xy_error = torch.zeros(n, device=device, dtype=torch.float64)
    history = None
    slip_distance = torch.zeros_like(xy_error)
    stance_time = torch.zeros_like(xy_error)
    measure_slip = runner.flat_depth_arm is not None and rough_seed is None
    if measure_slip:
      foot_ids, _ = motion.robot.find_sites(["left_foot", "right_foot"])
      if len(foot_ids) != 2:
        raise ValueError("Expected both sole sites for the stance-slip metric")
      previous_feet = None
    for t in range(max(frames)):
      if measure_slip:
        feet = motion.robot.data.site_pos_w[:, foot_ids]
        if previous_feet is not None:
          stance = (feet[..., 2] < .02) & (previous_feet[..., 2] < .02) & active[:, None]
          slip_distance += ((feet[..., :2] - previous_feet[..., :2]).norm(dim=-1) * stance).sum(1)
          stance_time += stance.sum(1) * sim.step_dt
        previous_feet = feet.clone()
      motion._update_metrics()
      xy = (motion.motion.root_pos_w[motion.time_steps, :2] + sim.scene.env_origins[:, :2]
            - sim.sim.data.qpos[:, :2]).norm(dim=-1)
      xy_error += torch.where(active, xy, 0.)
      for key in metrics:
        totals[key] += torch.where(active, motion.metrics[key], 0.0)
      rep = world.representation.represent(obs["full_state"])
      aux = representation.extract_aux_state(obs["full_state"])
      if runner.pelvis_motion_goal:
        _, goal = representation.compute_wm_goal(motion.motion, motion.time_steps, aux)
      else:
        goal = extract_actor_goal(obs["goal"])
      sensor = (rep if runner.actor_obs == "rep_state"
                else representation.append_pelvis_sensor(obs["actor"], rep))
      if history is None:
        history = sensor[:, None].repeat(1, runner.history_length, 1)
      else:
        history = torch.cat((history[:, 1:], sensor[:, None]), 1)
      td = build_actor_obs_td({"motion": goal}, history)
      if camera_cfg is not None:
        data = camera.data
        memory.update(data.distances, data.pos_w, data.quat_w)
        patch.bind(*patch.capture(memory))
        aux = representation.extract_aux_state(obs["full_state"])
        td["terrain"] = patch.features(aux)
        td["depth_tokens"] = patch.tokens(aux)
        td["future_feet"] = future_feet(motion.time_steps)
        if vision is not None:
          td["camera_latent"] = vision.encode(data.distances)
      elif runner.terrain_inputs != "blind":
        td["terrain"] = (ground.features(representation.extract_aux_state(obs["full_state"]))
                         if rough_seed is not None else rep.new_zeros(n, TerrainObservation.num_features))
      if runner.use_root_waypoints:
        td["root_waypoints"] = root_waypoints(obs["goal"])
      action = torch.where(active[:, None], actor(td), 0.0)
      obs, reward, terminated, truncated, _ = sim.step(action)
      ended = sim.termination_manager.get_term("motion_end")
      done = terminated | truncated
      if camera_cfg is not None:
        memory.reset(done)
      for name, mask in reasons.items():
        mask |= active & done & sim.termination_manager.get_term(name)
      if (active & truncated & ~ended & ~terminated).any():
        raise RuntimeError("Unexpected timeout during a flat regression clip")
      totals["return"] += torch.where(active, reward, 0.0)
      counts += active.long()
      completed |= active & ended & ~terminated
      active &= ~done
      if t % 1000 == 0:
        print(f"Frame {t}, {active.sum().item()} remaining", flush=True)
      if not active.any():
        break
    if active.any():
      raise RuntimeError("At least one reference did not reach an episode boundary")
    rows = [
      dict(
        motion=motion.motion.motion_names[i],
        planned_frames=frames[i],
        steps=int(counts[i]),
        completed=bool(completed[i]),
        mean_pelvis_xy_error_m=float(xy_error[i] / counts[i]),
        termination_terms=[name for name, mask in reasons.items() if mask[i]],
        reward_per_reference_frame=float(totals["return"][i] / frames[i]),
        **{k: float(totals[k][i] / counts[i]) for k in metrics},
      )
      for i in range(n)
    ]
    if measure_slip:
      for i, row in enumerate(rows):
        row["stance_sole_speed_m_s"] = float(slip_distance[i] / stance_time[i]) if stance_time[i] > 0 else None
        row["stance_foot_seconds"] = float(stance_time[i])
    return dict(
      checkpoint=str(checkpoint),
      actor_sha256=hashlib.sha256((checkpoint / "actor.pt").read_bytes()).hexdigest(),
      depth_input="noiseless_pelvis_camera" if camera_cfg is not None else None,
      completed=int(completed.sum()),
      motions=n,
      terrain_seed=rough_seed,
      starts_sha256=hashlib.sha256(starts_file.read_bytes()).hexdigest(),
      mean_pelvis_xy_error_m=float((xy_error / counts).mean()),
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
  p.add_argument("--rough-seed", type=int, help="Evaluate all clips on an irregular 9/4 cm field")
  p.add_argument("--save-starts", action="store_true", help="Create a new paired start snapshot")
  a = p.parse_args()
  result = evaluate(
    a.checkpoint, json.loads(a.protocol.read_text()), a.starts, a.device,
    rough_seed=a.rough_seed, save_starts=a.save_starts,
  )
  a.output.parent.mkdir(exist_ok=True, parents=True)
  a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(result["completed"], result["means"])
