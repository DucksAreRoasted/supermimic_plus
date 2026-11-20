"""Paired G1 walking windows in MuJoCo, with open-loop world-model error on real actions."""

import argparse
import hashlib
import json
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.representations import extract_actor_goal, root_waypoints
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.walk_terrain import TerrainObservation, WalkTerrain


def checkpoint_config(path):
  meta = json.loads((path / "run_manifest.json").read_text())
  if "runner_cfg" in meta:
    return load_runner_cfg(meta)
  if (
    meta["run_name"]
    != "g1-shared-focus04-nograph-reset-safe-simweight0-warm24k-plus16k-seed42"
  ):
    raise ValueError("Archived run needs an explicit architecture")
  return SuperMimicRunnerCfg(residual_correction=True, residual_shared_trunk=True)


def select_windows(root, heights, length, count, seed):
  # Each test window commands actual travel and encounters a change of elevation.
  candidates = [
    t
    for t in range(0, len(root) - length - 1, 10)
    if np.linalg.norm(root[t + length, :2] - root[t, :2]) > 0.75
    and np.ptp(heights[t : t + length + 1]) >= 0.02
  ]
  if len(candidates) < count:
    raise ValueError(
      f"Only {len(candidates)} obstacle-crossing starts for {count} slots"
    )
  return sorted(
    np.random.default_rng(seed).choice(candidates, count, replace=False).tolist()
  )


@torch.inference_mode()
def evaluate(args):
  device = args.device
  course = WalkTerrain(args.height, args.phase, lateral_height=args.lateral_height)
  query = TerrainObservation(course).to(device)
  with np.load(args.motion) as clip:
    root = clip["root_pos_w"]
    heights = query.height_at(torch.as_tensor(root[:, :2], device=device)).cpu().numpy()
    starts = select_windows(root, heights, args.steps, args.num_envs, args.seed)
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  cfg.seed = args.seed
  cfg.scene.env_spacing = 0.0
  cfg.scene.spec_fn = WalkTerrain(0.0) if args.flat else course
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 512)
  if args.solver_iterations:
    cfg.sim.mujoco.iterations = args.solver_iterations
    cfg.sim.mujoco.ls_iterations = args.solver_iterations
  cfg.events = {
    k: v
    for k, v in cfg.events.items()
    if k in ("foot_friction", "base_com", "encoder_bias")
  }
  cmd = cfg.commands["motion"]
  cmd.motion_file = str(args.motion)
  cmd.sampling_mode = "start"
  cmd.pose_range, cmd.velocity_range, cmd.joint_position_range = {}, {}, (0.0, 0.0)
  env = create_env(cfg, args.num_envs, device)
  try:
    sim = env.unwrapped
    model_cfg = checkpoint_config(args.checkpoint)
    wm = build_world_model(model_cfg, env, device)
    wm.load_bundle(
      torch.load(
        args.checkpoint / "world_model.pt", map_location=device, weights_only=True
      )
    )
    wm.dynamics.eval()
    models = {str(args.checkpoint): wm}
    for path in args.compare_world_models:
      other = build_world_model(checkpoint_config(path), env, device)
      other.load_bundle(
        torch.load(path / "world_model.pt", map_location=device, weights_only=True)
      )
      other.dynamics.eval()
      models[str(path)] = other
    actor_cfg = checkpoint_config(args.driver or args.checkpoint)
    actor = build_actor(actor_cfg, wm, device)
    actor.load_state_dict(
      torch.load(
        (args.driver or args.checkpoint) / "actor.pt",
        map_location=device,
        weights_only=True,
      )
    )
    actor.eval()
    sim.reset(seed=args.seed)
    m = sim.command_manager.get_term("motion")
    for i, t in enumerate(starts):
      m.reset_to_frame(torch.tensor([i], device=device), t)
    sim.scene.write_data_to_sim()
    sim.sim.forward()
    sim.sim.sense()
    m.update_relative_body_poses()
    obs = sim.observation_manager.compute(update_history=True)
    assert m.time_steps.tolist() == starts
    ctx = wm.sim_context.read(sim)
    start_diffs = {}
    if args.starts_file:
      snapshot = dict(
        frames=np.asarray(starts),
        qpos=sim.sim.data.qpos.cpu().numpy(),
        qvel=sim.sim.data.qvel.cpu().numpy(),
        context=ctx.cpu().numpy(),
        full_state=obs["full_state"].cpu().numpy(),
      )
      if args.starts_file.exists():
        with np.load(args.starts_file) as reference:
          for name, array in snapshot.items():
            start_diffs[name] = float(np.max(np.abs(array - reference[name])))
            if name in ("frames", "qpos", "qvel"):
              np.testing.assert_array_equal(array, reference[name], err_msg=name)
            else:
              np.testing.assert_allclose(
                array, reference[name], rtol=0, atol=1.0e-6, err_msg=name
              )
      else:
        args.starts_file.parent.mkdir(parents=True, exist_ok=True)
        with args.starts_file.open("xb") as dest:
          np.savez(dest, **snapshot)
    start_sha = hashlib.sha256(
      torch.cat([obs["full_state"], ctx], -1).cpu().numpy().tobytes()
    ).hexdigest()
    rep = wm.representation
    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    returns = torch.zeros_like(active, dtype=torch.float64)
    steps = torch.zeros_like(active, dtype=torch.long)
    body_error = torch.zeros_like(returns)
    root_error = torch.zeros_like(returns)
    floor0 = query.height_at(obs["full_state"][:, :2])
    low, high = floor0, floor0
    wm_metrics = {name: {h: [] for h in (1, 8, 32)} for name in models}
    ankles = [m.cfg.body_names.index(s + "_ankle_roll_link") for s in ("left", "right")]
    for t in range(args.steps):
      aux = rep.extract_aux_state(obs["full_state"])
      state = rep.represent(obs["full_state"])
      if t % 32 == 0:
        predictions = dict.fromkeys(models, (aux, state))
      td = build_actor_obs_td(
        {"motion": extract_actor_goal(obs["goal"])}, state.unsqueeze(1)
      )
      if actor_cfg.terrain_inputs != "blind":
        td["terrain"] = (
          torch.zeros_like(query.features(aux)) if args.flat else query.features(aux)
        )
      if actor_cfg.use_root_waypoints:
        td["root_waypoints"] = root_waypoints(obs["goal"])
      actions = actor(td)
      for name, model in models.items():
        pred_aux, pred_state = predictions[name]
        predictions[name] = model.dynamics(pred_state, actions, pred_aux, ctx)
      m._update_metrics()
      body_error += torch.where(active, m.metrics["error_body_pos"], 0.0)
      obs, r, done, _ = env.step(actions)
      returns += torch.where(active, r, 0.0)
      steps += active.long()
      active = active & ~done.bool()
      real = rep.extract_aux_state(obs["full_state"])
      root_error = torch.where(
        active, (real[:, :2] - m.anchor_pos_w[:, :2]).norm(dim=-1), root_error
      )
      height = query.height_at(real[:, :2])
      low = torch.where(active, torch.minimum(low, height), low)
      high = torch.where(active, torch.maximum(high, height), high)
      h = t % 32 + 1
      if h not in (1, 8, 32) or not active.any():
        continue
      for name, (pred_aux, pred_state) in predictions.items():
        feet = rep.world_frame_bodies(pred_aux, pred_state)[0][:, ankles]
        ankle_err = (feet - m.robot_body_pos_w[:, ankles]).norm(dim=-1)
        z = m.robot_body_pos_w[:, ankles, 2]
        floor = (
          torch.zeros_like(z)
          if args.flat
          else query.height_at(m.robot_body_pos_w[:, ankles, :2])
        )
        near = (z - floor < 0.075) & active[:, None]
        wm_metrics[name][h].append(
          torch.stack(
            (
              (pred_aux[:, :3] - real[:, :3]).norm(dim=-1)[active].sum(),
              active.sum(),
              ankle_err[near].sum(),
              near.sum(),
            )
          )
        )
    success = active & (root_error <= 0.4)
    if not args.flat:
      success &= (high - low) >= 0.02
    rows = [
      dict(
        start=t,
        steps=int(steps[i]),
        survived=bool(active[i]),
        success=bool(success[i]),
        reward_per_planned_step=float(returns[i] / args.steps),
        mean_body_error_cm=float(100 * body_error[i] / steps[i].clamp_min(1)),
        last_root_xy_error_m=float(root_error[i]),
        crossed_height_m=float(high[i] - low[i]),
      )
      for i, t in enumerate(starts)
    ]
    report = dict(
      checkpoint=str(args.checkpoint),
      driver=str(args.driver or args.checkpoint),
      clip_sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
      height=args.height,
      lateral_height=args.lateral_height,
      phase_cells=args.phase,
      flat=args.flat,
      solver_iterations=cfg.sim.mujoco.iterations,
      solver_ls_iterations=cfg.sim.mujoco.ls_iterations,
      seed=args.seed,
      steps=args.steps,
      start_state_sha256=start_sha,
      initial_max_abs_difference=start_diffs,
      survived=int(active.sum()),
      successes=int(success.sum()),
      windows=len(rows),
      mean_reward_per_planned_step=float(returns.mean() / args.steps),
      mean_body_error_cm=float(np.mean([r["mean_body_error_cm"] for r in rows])),
      windows_detail=rows,
    )
    report["comparison_open_loop"] = {}
    for name, by_horizon in wm_metrics.items():
      results = {}
      for h, batches in by_horizon.items():
        if batches:
          root_sum, n, foot_sum, n_foot = torch.stack(batches).sum(0).tolist()
          results[h] = dict(
            root_cm=100 * root_sum / n,
            samples=int(n),
            near_ground_ankle_cm=100 * foot_sum / n_foot if n_foot else None,
            near_ground_samples=int(n_foot),
          )
      report["comparison_open_loop"][name] = results
    report["open_loop"] = report["comparison_open_loop"][str(args.checkpoint)]
    return report
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("motion", type=Path)
  parser.add_argument("out", type=Path)
  parser.add_argument(
    "--driver",
    type=Path,
    help="Collect all real actions with a fixed comparison policy",
  )
  parser.add_argument(
    "--compare-world-models",
    type=Path,
    nargs="+",
    default=[],
    help="Score additional dynamics on exactly the same simulated transitions",
  )
  parser.add_argument("--height", type=float, default=0.065)
  parser.add_argument("--phase", type=int, default=17)
  parser.add_argument("--lateral-height", type=float, default=0.0)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seed", type=int, default=71)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--flat", action="store_true")
  parser.add_argument(
    "--solver-iterations", type=int, help="Set both solver iteration limits"
  )
  parser.add_argument(
    "--starts-file", type=Path, help="Record or verify the common initial conditions"
  )
  args = parser.parse_args()
  result = evaluate(args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(
    json.dumps({k: v for k, v in result.items() if k != "windows_detail"}, indent=2)
  )


if __name__ == "__main__":
  main()
