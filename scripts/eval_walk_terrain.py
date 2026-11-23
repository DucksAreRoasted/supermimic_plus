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
from supermimic.tasks.depth_mimic import (
  DepthMemory,
  DepthTrackingController,
  walking_camera,
)
from supermimic.tasks.foot_depth_actor import (
  FutureFeet,
  MeasuredPatch,
  file_hash,
  load_fusion,
)
from supermimic.tasks.perceptive_walk import (
  LocalGroundScan,
  TerrainMotionWindow,
  walking_scan,
)
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.walk_terrain import (
  IrregularWalkTerrain,
  TerrainObservation,
  WalkTerrain,
)


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


def select_windows(root, heights, length, count, seed, *, disjoint=False):
  # Each test window commands actual travel and encounters a change of elevation.
  candidates = [
    t
    for t in range(0, len(root) - length - 1, 10)
    if np.linalg.norm(root[t + length, :2] - root[t, :2]) > 0.75
    and np.ptp(heights[t : t + length + 1]) >= 0.02
  ]
  if disjoint:
    independent = []
    for frame in candidates:
      if not independent or frame > independent[-1] + length:
        independent.append(frame)
    candidates = independent
  if len(candidates) < count:
    raise ValueError(
      f"Only {len(candidates)} obstacle-crossing starts for {count} slots"
    )
  return sorted(
    np.random.default_rng(seed).choice(candidates, count, replace=False).tolist()
  )


def render_walk(path, model, poses, reference, dt):
  """Render a simulated window with the unmodified reference in green."""
  import imageio.v2 as imageio
  import mujoco

  actual, ghost = mujoco.MjData(model), mujoco.MjData(model)
  camera = mujoco.MjvCamera()
  travel = reference[-1, :2] - reference[0, :2]
  camera.azimuth = np.degrees(np.arctan2(travel[1], travel[0])) + 74
  camera.distance = 3.25
  camera.elevation = -14
  options, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
  path.parent.mkdir(parents=True, exist_ok=True)
  with (
    mujoco.Renderer(model, height=384, width=640) as renderer,
    imageio.get_writer(path, fps=1 / (2 * dt), codec="libx264", quality=8) as writer,
  ):
    for pose, ref in zip(poses, reference, strict=True):
      actual.qpos[:], ghost.qpos[:] = pose, ref
      mujoco.mj_forward(model, actual)
      mujoco.mj_forward(model, ghost)
      camera.lookat[:] = (pose[:3] + ref[:3]) / 2
      renderer.update_scene(actual, camera, scene_option=options)
      scene = renderer.scene
      begin = scene.ngeom
      mujoco.mjv_addGeoms(
        model, ghost, options, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene
      )
      for idx in range(begin, scene.ngeom):
        geom = scene.geoms[idx]
        geom.matid = -1
        geom.rgba[:] = (0.10, 0.95, 0.32, 0.40)
      writer.append_data(renderer.render())


@torch.inference_mode()
def evaluate(args):
  device = args.device
  actor_cfg = checkpoint_config(args.driver or args.checkpoint)
  embedded = actor_cfg.depth_fusion != "none"
  if embedded and args.fusion_checkpoint:
    raise ValueError("This Actor already includes its trained depth adapter")
  learned_fusion = embedded or args.fusion_checkpoint is not None
  fusion_eval = learned_fusion or args.record_fusion_bank
  if fusion_eval and (args.perception != "depth" or args.stride_gain):
    raise ValueError("Learned fusion requires raw depth and unmodified motion")
  course = WalkTerrain(args.height, args.phase, lateral_height=args.lateral_height)
  if args.landscape_seed is not None:
    course = IrregularWalkTerrain(
      height=args.height,
      lateral_height=args.lateral_height,
      landscape_seed=args.landscape_seed,
    )
  query = TerrainObservation(course).to(device)
  with np.load(args.motion) as clip:
    root = clip["root_pos_w"][args.first_start_frame : args.window_search_frames]
    heights = query.height_at(torch.as_tensor(root[:, :2], device=device)).cpu().numpy()
    starts = [
      args.first_start_frame + t
      for t in select_windows(
        root,
        heights,
        args.steps,
        args.num_envs,
        args.seed,
        disjoint=args.disjoint_windows,
      )
    ]
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  cfg.seed = args.seed
  cfg.scene.env_spacing = 0.0
  cfg.scene.spec_fn = WalkTerrain(0.0) if args.flat else course
  sensed = args.perception != "map"
  if sensed:
    cfg.scene.sensors = (*cfg.scene.sensors, walking_scan())
  depth = args.perception.startswith("depth")
  if depth:
    camera_cfg = walking_camera()
    cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg)
  cfg.sim.nconmax = max(
    cfg.sim.nconmax or 0, 256 if args.landscape_seed is not None else 128
  )
  cfg.sim.njmax = max(
    cfg.sim.njmax or 0, 1024 if args.landscape_seed is not None else 512
  )
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
    measured_models = {str(args.checkpoint)} if model_cfg.depth_fusion != "none" else set()
    for path in args.compare_world_models:
      other_cfg = checkpoint_config(path)
      if other_cfg.depth_fusion != "none":
        measured_models.add(str(path))
      other = build_world_model(other_cfg, env, device)
      other.load_bundle(
        torch.load(path / "world_model.pt", map_location=device, weights_only=True)
      )
      other.dynamics.eval()
      models[str(path)] = other
    driver_wm = wm
    if actor_cfg.imagined_vision != "none" and args.driver and args.driver != args.checkpoint:
      driver_wm = build_world_model(actor_cfg, env, device)
      driver_wm.load_bundle(torch.load(args.driver / "world_model.pt", map_location=device, weights_only=True))
    actor = build_actor(actor_cfg, driver_wm, device)
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
    if depth:
      camera = sim.scene["walk_depth"]
      memory = DepthMemory(args.num_envs, device, sim.step_dt, camera_cfg.pattern)
      camera_rng = torch.Generator(device=device).manual_seed(args.seed + 8731)
      controller = DepthTrackingController(
        m,
        wm,
        memory,
        args.perception,
        foot_gain=args.foot_gain,
        steering_gain=args.steering_gain,
        balance_gain=args.balance_gain,
        prediction_steps=args.prediction_steps,
        stride_gain=args.stride_gain,
      )
    if measured_models and not fusion_eval:
      raise ValueError("Measuring online depth dynamics requires a depth-equipped driver")
    if measured_models:
      prediction_patch = MeasuredPatch(query.offsets)
      for name in measured_models:
        models[name].dynamics.terrain_query = prediction_patch
    if fusion_eval:
      patch = MeasuredPatch(query.offsets)
      previous = torch.zeros(args.num_envs, env.num_actions, device=device)
      records = []
      if learned_fusion:
        reference_feet = FutureFeet(m)
        fusion_mode = actor_cfg.depth_fusion if embedded else load_fusion(
          actor,
          args.fusion_checkpoint,
          (args.driver or args.checkpoint) / "actor.pt",
          m.motion.fk.joint_names,
          device,
        )
    for i, t in enumerate(starts):
      m.reset_to_frame(torch.tensor([i], device=device), t)
    sim.scene.write_data_to_sim()
    sim.sim.forward()
    sim.sim.sense()
    if sensed:
      sensor = sim.scene["walk_scan"]
      cloud = LocalGroundScan(sensor.data)
      center = sim.sim.data.qpos.new_zeros((args.num_envs, 1, 2))
      lift, valid = cloud.at(center)
      if not valid.all():
        raise ValueError("Missing terrain underneath a starting pelvis")
      sim.sim.data.qpos[:, 2] += lift[:, 0]
      sim.sim.forward()
      sim.sim.sense()
    adapter = (
      TerrainMotionWindow(m, sim.step_dt) if args.perception == "online" else None
    )
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
    body_trace = []
    upper_error = torch.zeros_like(returns)
    scan_totals = {
      name: returns.new_zeros(())
      for name in ("error", "visible", "points", "bend", "coverage", "frames")
    }
    fusion_deltas = torch.zeros(3, device=device, dtype=torch.float64)
    initial_scan_action_delta = None
    root_error = torch.zeros_like(returns)
    if learned_fusion:
      start_ref_xy = m.anchor_pos_w[:, :2].clone()
      last_ref_xy = start_ref_xy.clone()
      last_robot_xy = obs["full_state"][:, :2].clone()
      termination_terms = {k: torch.zeros_like(active) for k in sim.termination_manager.active_terms}
    floor0 = query.height_at(obs["full_state"][:, :2])
    low, high = floor0, floor0
    prediction_horizons = (1, 8, 16, 32)
    wm_metrics = {name: {h: [] for h in prediction_horizons} for name in models}
    ankles = [m.cfg.body_names.index(s + "_ankle_roll_link") for s in ("left", "right")]
    upper = [
      i
      for i, name in enumerate(m.cfg.body_names)
      if any(part in name for part in ("torso", "shoulder", "elbow", "wrist"))
    ]
    offsets = query.offsets.expand(args.num_envs, -1, -1)
    video_poses = []
    for t in range(args.steps):
      if args.video and t % 2 == 0:
        video_poses.append(sim.sim.data.qpos.clone())
      aux = rep.extract_aux_state(obs["full_state"])
      state = rep.represent(obs["full_state"])
      if t % 32 == 0:
        predictions = dict.fromkeys(models, (aux, state))
      goal = extract_actor_goal(obs["goal"])
      if sensed:
        if depth:
          data = camera.data
          ranges = data.distances
          if args.depth_noise_m:
            noise = torch.randn(ranges.shape, device=device, generator=camera_rng)
            ranges = torch.where(
              ranges > 0, ranges + args.depth_noise_m * noise, ranges
            )
          if args.depth_dropout:
            missing = (
              torch.rand(ranges.shape, device=device, generator=camera_rng)
              < args.depth_dropout
            )
            ranges = ranges.masked_fill(missing, -1.0)
          terrain, heights, valid = controller.observe(
            ranges, data.pos_w, data.quat_w, offsets
          )
          if fusion_eval:
            image, origin = patch.capture(memory)
            patch.bind(image, origin)
            if measured_models and t % 32 == 0:
              prediction_patch.bind(image, origin)
            terrain = patch.features(aux)
            if args.record_fusion_bank and t % 6 == 0 and active.any():
              records.append(
                {
                  key: val[active].cpu()
                  for key, val in dict(
                    image=image,
                    origin=origin,
                    state=obs["full_state"],
                    motion_time=m.time_steps,
                    sim_ctx=ctx,
                    prev_action=previous,
                  ).items()
                }
              )
          else:
            goal = controller.goal(goal, aux)
        else:
          scan = LocalGroundScan(sensor.data)
          heights, valid = scan.at(offsets)
          terrain = (
            torch.zeros_like(heights) if args.perception == "blind" else 10 * heights
          )
        # Compare measured and known heights for scoring only.
        visible = valid & active[:, None]
        expected = torch.zeros_like(heights) if args.flat else query(aux)
        scan_totals["error"] += ((heights - expected).abs() * visible).sum()
        scan_totals["visible"] += visible.sum()
        scan_totals["points"] += active.sum() * heights.shape[1]
        if adapter is not None:
          goal, bend, coverage = adapter.actor_goal(scan, m.time_steps, goal)
          scan_totals["bend"] += (bend * active[:, None]).sum()
          scan_totals["coverage"] += (coverage * active[:, None]).sum()
          scan_totals["frames"] += active.sum() * bend.shape[1]
      td = build_actor_obs_td({"motion": goal}, state.unsqueeze(1))
      if actor_cfg.terrain_inputs != "blind":
        td["terrain"] = (
          terrain
          if sensed
          else (
            torch.zeros_like(query.features(aux)) if args.flat else query.features(aux)
          )
        )
      if actor_cfg.use_root_waypoints:
        td["root_waypoints"] = root_waypoints(obs["goal"])
      if learned_fusion:
        td["depth_tokens"] = patch.tokens(aux)
        td["future_feet"] = reference_feet(m.time_steps)
      if actor_cfg.imagined_vision != "none":
        if not depth:
          raise ValueError("The visual policy needs its real depth sensor")
        td["camera_latent"] = driver_wm.dynamics.vision.encode(ranges)
      actions = actor(td)
      if learned_fusion and t % 32 == 0:
        head = actor.foot_depth_fusion
        base = actor.decode_actions(td)
        masked = td["depth_tokens"].clone()
        masked[..., 2:] = 0
        blind = head(base, masked, td["future_feet"])
        if actor.visual_action is not None:
          blind = blind + actor.visual_action(td["camera_latent"])
        leg = head.legs.flatten()
        fusion_deltas[0] += ((actions - base)[:, leg].abs().mean(-1) * active).sum()
        fusion_deltas[1] += ((actions - blind)[:, leg].abs().mean(-1) * active).sum()
        fusion_deltas[2] += active.sum()
      if fusion_eval:
        previous = actions.detach()
      elif depth:
        actions = controller.action(actions, aux, state, terrain)
      if (
        t == 0
        and args.perception in ("scan", "online")
        and actor_cfg.terrain_inputs != "blind"
      ):
        td["terrain"] = torch.zeros_like(terrain)
        initial_scan_action_delta = float((actions - actor(td)).abs().mean())
      for name, model in models.items():
        pred_aux, pred_state = predictions[name]
        predictions[name] = model.dynamics(pred_state, actions, pred_aux, ctx)
      m._update_metrics()
      body_error += torch.where(active, m.metrics["error_body_pos"], 0.0)
      if learned_fusion:
        body_trace.append(m.metrics["error_body_pos"].detach().clone())
      if sensed:
        upper_distance = (
          m.body_pos_relative_w[:, upper] - m.robot_body_pos_w[:, upper]
        ).norm(dim=-1)
        upper_error += torch.where(active, upper_distance.mean(-1), 0.0)
      obs, r, done, _ = env.step(actions)
      if depth:
        controller.reset(done.bool())
      returns += torch.where(active, r, 0.0)
      steps += active.long()
      if learned_fusion:
        first_done = active & done.bool()
        for name, occurred in termination_terms.items():
          occurred |= first_done & sim.termination_manager.get_term(name).bool()
      active = active & ~done.bool()
      real = rep.extract_aux_state(obs["full_state"])
      root_error = torch.where(
        active, (real[:, :2] - m.anchor_pos_w[:, :2]).norm(dim=-1), root_error
      )
      if learned_fusion:
        last_ref_xy = torch.where(active[:, None], m.anchor_pos_w[:, :2], last_ref_xy)
        last_robot_xy = torch.where(active[:, None], real[:, :2], last_robot_xy)
      height = query.height_at(real[:, :2])
      low = torch.where(active, torch.minimum(low, height), low)
      high = torch.where(active, torch.maximum(high, height), high)
      h = t % 32 + 1
      if h not in prediction_horizons or not active.any():
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
    if args.record_fusion_bank:
      args.record_fusion_bank.parent.mkdir(parents=True, exist_ok=True)
      torch.save(
        dict(
          motion=str(args.motion.resolve()),
          clip_sha256=file_hash(args.motion),
          base_sha256=file_hash((args.driver or args.checkpoint) / "actor.pt"),
          frames=len(m.motion.joint_pos),
          tensors={k: torch.cat([r[k] for r in records]) for k in records[0]},
        ),
        args.record_fusion_bank,
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
    if learned_fusion:
      displacement = last_ref_xy - start_ref_xy
      distance = displacement.norm(dim=-1)
      forward_xy = displacement / distance[:, None].clamp_min(1.e-6)
      offset = last_robot_xy - last_ref_xy
      along = (forward_xy * offset).sum(-1)
      lateral = forward_xy[:, 0] * offset[:, 1] - forward_xy[:, 1] * offset[:, 0]
      body_trace_cm = (100 * torch.stack(body_trace, dim=1)).cpu()
      for i, row in enumerate(rows):
        row["termination_terms"] = [name for name, occurred in termination_terms.items() if occurred[i]]
        row["last_ref_displacement_m"] = float(distance[i])
        row["last_along_error_m"] = float(along[i]) if distance[i] > 0.10 else None
        row["last_lateral_error_m"] = float(lateral[i]) if distance[i] > 0.10 else None
        row["body_error_trace_cm"] = body_trace_cm[i, : row["steps"]].tolist()
        row["upper_body_error_cm"] = float(100 * upper_error[i] / row["steps"])
    report = dict(
      checkpoint=str(args.checkpoint),
      driver=str(args.driver or args.checkpoint),
      clip_sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
      height=args.height,
      lateral_height=args.lateral_height,
      landscape_seed=args.landscape_seed,
      phase_cells=args.phase,
      flat=args.flat,
      perception=args.perception,
      fusion_mode=fusion_mode if learned_fusion else None,
      fusion_sha256=file_hash(args.fusion_checkpoint or ((args.driver or args.checkpoint) / "actor.pt"))
      if learned_fusion else None,
      external_joint_compensation=bool(
        depth
        and not fusion_eval
        and (controller.use_balance or controller.predictor is not None)
      ),
      stride_gain=controller.stride_gain if depth else 0.0,
      depth_noise_m=args.depth_noise_m,
      depth_dropout=args.depth_dropout,
      prediction_steps=args.prediction_steps
      if args.perception.startswith("depth-predict")
      else None,
      balance_gain=args.balance_gain if depth else None,
      steering_gain=args.steering_gain if depth else None,
      foot_gain=args.foot_gain if args.perception.startswith("depth-contact") else None,
      window_search_frames=args.window_search_frames,
      first_start_frame=args.first_start_frame,
      disjoint_windows=args.disjoint_windows,
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
    if learned_fusion:
      total, masked, count = fusion_deltas.tolist()
      report["mean_leg_action_delta_from_base"] = total / count
      report["mean_leg_action_delta_masking_fusion_depth"] = masked / count
    if sensed:
      totals = {k: float(v) for k, v in scan_totals.items()}
      report["upper_body_links"] = [m.cfg.body_names[i] for i in upper]
      report["mean_upper_body_error_cm"] = float(
        (100 * upper_error / steps.clamp_min(1)).mean()
      )
      report["scan"] = dict(
        valid_fraction=totals["visible"] / totals["points"],
        height_mae_cm=100 * totals["error"] / totals["visible"]
        if totals["visible"]
        else None,
        initial_action_mae_vs_masked_scan=initial_scan_action_delta,
      )
      if totals["frames"]:
        report["scan"].update(
          mean_leg_bend_rad=totals["bend"] / totals["frames"],
          footprint_coverage=totals["coverage"] / totals["frames"],
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
    if args.video:
      if args.video_first_success:
        passed = [i for i, r in enumerate(rows) if r["success"]]
        if not passed:
          report["video"] = {"path": None, "reason": "no successful windows"}
          return report
        index = passed[0]
      else:
        index = 0 if args.video_start is None else starts.index(args.video_start)
      length = (int(steps[index]) + 1) // 2
      poses = torch.stack([batch[index] for batch in video_poses[:length]])
      motion = m.motion
      frames = starts[index] + 2 * torch.arange(length, device=device)
      reference = poses.clone()
      reference[:, :3] = motion.root_pos_w[frames]
      reference[:, 3:7] = motion.root_quat_w[frames]
      reference[:, m.robot.indexing.joint_q_adr] = motion.joint_pos[frames]
      render_walk(
        args.video,
        sim.sim.mj_model,
        poses.cpu().numpy(),
        reference.cpu().numpy(),
        sim.step_dt,
      )
      report["video"] = {
        "start": starts[index],
        "frames": length,
        "path": str(args.video),
      }
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
  parser.add_argument(
    "--landscape-seed",
    type=int,
    help="Evaluate on an independently generated irregular 2-D surface",
  )
  parser.add_argument("--height", type=float, default=0.065)
  parser.add_argument("--phase", type=int, default=17)
  parser.add_argument("--lateral-height", type=float, default=0.0)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument(
    "--window-search-frames",
    type=int,
    help="Select starts within the first N original frames",
  )
  parser.add_argument("--seed", type=int, default=71)
  parser.add_argument("--first-start-frame", type=int, default=0)
  parser.add_argument("--disjoint-windows", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--video", type=Path, help="Render a paired window with its raw reference"
  )
  video_selection = parser.add_mutually_exclusive_group()
  video_selection.add_argument(
    "--video-start", type=int, help="Choose a recorded start frame"
  )
  video_selection.add_argument(
    "--video-first-success",
    action="store_true",
    help="Illustrate the first passing window in this run",
  )
  parser.add_argument("--flat", action="store_true")
  parser.add_argument("--record-fusion-bank", type=Path)
  parser.add_argument("--fusion-checkpoint", type=Path)
  parser.add_argument(
    "--perception",
    choices=(
      "map",
      "blind",
      "scan",
      "online",
      "depth",
      "depth-blind",
      "depth-balance",
      "depth-predict",
      "depth-predict-no-depth",
      "depth-steer",
      "depth-contact",
      "depth-contact-shuffled",
      "depth-proprio-stride",
      "depth-contact-stride",
      "depth-contact-stride-shuffled",
      "depth-contact-recover",
      "depth-contact-recover-shuffled",
      "depth-contact-budget",
      "depth-contact-budget-shuffled",
      "depth-steer-blind",
      "depth-steer-always",
    ),
    default="map",
    help="Map reproduces the analytic-map protocol. Depth-contact runs the pinhole camera, reference steering and per-foot corrections on the raw motion.",
  )
  parser.add_argument(
    "--stride-gain", type=float, help="Override the phase-specific hip excursion gain"
  )
  parser.add_argument("--foot-gain", type=float, default=1.5)
  parser.add_argument("--steering-gain", type=float, default=0.45)
  parser.add_argument("--prediction-steps", type=int, choices=range(1, 9), default=4)
  parser.add_argument("--balance-gain", type=float, default=0.5)
  parser.add_argument(
    "--solver-iterations", type=int, help="Set both solver iteration limits"
  )
  parser.add_argument(
    "--starts-file", type=Path, help="Record or verify the common initial conditions"
  )
  parser.add_argument("--depth-noise-m", type=float, default=0.0)
  parser.add_argument("--depth-dropout", type=float, default=0.0)
  args = parser.parse_args()
  if args.depth_noise_m < 0 or not 0 <= args.depth_dropout <= 1:
    parser.error("Depth noise must be nonnegative and dropout must lie in [0,1]")
  result = evaluate(args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(
    json.dumps({k: v for k, v in result.items() if k != "windows_detail"}, indent=2)
  )


if __name__ == "__main__":
  main()
