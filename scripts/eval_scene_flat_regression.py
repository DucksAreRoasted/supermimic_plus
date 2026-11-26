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
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

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
def evaluate(checkpoint, protocol, starts_file, device, *, rough_seed=None, save_starts=False,
             rough_scale=1.0, leg_ablation=None, base_terrain=None,
             camera_diagnostic=None, max_steps=None, rollout_probe=None,
             direct_reference=False, action_controller=None):
  if direct_reference and rough_seed is None:
    raise ValueError("A generated terrain reference requires its fixed terrain seed")
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
  runner = checkpoint_config(checkpoint)
  camera_cfg = walking_camera(runner.camera_profile) if runner.depth_fusion != "none" else None
  if camera_diagnostic is not None and (camera_cfg is None or runner.camera_profile != "instinct"):
    raise ValueError("Camera diagnostics require an Instinct camera checkpoint")
  if action_controller is not None and camera_cfg is None:
    raise ValueError("The action controller requires measured depth")
  if camera_cfg is not None or rough_seed is not None:
    cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg or walking_camera())
  if rollout_probe is not None:
    rollout_probe.configure(cfg)
  clips = protocol["clips"]
  for clip in clips:
    with open(clip["path"], "rb") as source:
      if hashlib.file_digest(source, "sha256").hexdigest() != clip["sha256"]:
        raise ValueError(f"Motion checksum differs for {clip['path']}")
  cfg.scene.spec_fn = WalkTerrain(0.0)
  if rough_seed is not None:
    cfg.scene.spec_fn = IrregularWalkTerrain(height=0.09 * rough_scale, lateral_height=0.04 * rough_scale, landscape_seed=rough_seed)
    cfg.scene.env_spacing = 0.0
    if not direct_reference:
      from supermimic.tasks.terrain_reset import TerrainResetMotionCommandCfg

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
      np.testing.assert_array_equal(motion.time_steps.cpu(), saved["frames"], err_msg="frames")
      for key, states in (("qpos", sim.sim.data.qpos), ("qvel", sim.sim.data.qvel)):
        # GPU reset arithmetic can vary by an ulp. Use the saved simulator state
        # so every arm still starts from exactly the same position and velocity.
        np.testing.assert_allclose(states.cpu(), saved[key], rtol=0, atol=1e-6, err_msg=key)
        states.copy_(torch.as_tensor(saved[key], device=device))
      sim.sim.forward()
      sim.sim.sense()
      motion.update_relative_body_poses()
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

      vision = DepthVision(runner.imagined_vision, representation.rep_state_dim, 29, joint=runner.joint_visual_dynamics, camera_profile=runner.camera_profile).to(device).eval()
      bundle = torch.load(checkpoint / "world_model.pt", map_location=device, weights_only=True)
      vision.load_state_dict({k.removeprefix("vision."): v for k, v in bundle["dynamics"].items()
                              if k.startswith("vision.")})
      del bundle
    n = len(clips)
    if base_terrain is not None and (camera_cfg is None or runner.terrain_inputs == "blind"):
      raise ValueError("Base terrain ablation requires the measured terrain input")
    motion_groups = ("forward", "lateral", "turn", "backward", "other")
    if base_terrain is not None:
      # Per-motion counts grouped by reference travel direction at each active step.
      terrain_counts = torch.zeros(n, len(motion_groups), 4, device=device, dtype=torch.long)
      group_ids = torch.arange(len(motion_groups), device=device)
      root_history = torch.empty(12, n, 7, device=device)
      group_history = torch.empty(12, n, device=device, dtype=torch.long)
      landing_counts = torch.zeros(n, len(motion_groups), 3, 2, device=device, dtype=torch.long)

    correction_sum = torch.zeros(n, device=device, dtype=torch.float64)
    if leg_ablation is not None:
      if actor.foot_depth_fusion is None:
        raise ValueError("Leg ablation requires a trained leg modulator")
      def measure_leg_delta(module, inputs, output):
        base = inputs[0]
        delta = (output - base)[:, module.legs.flatten()]
        if leg_ablation == "off":
          return base
        correction_sum.add_(delta.abs().mean(-1) * active)
        return output
      actor.foot_depth_fusion.register_forward_hook(measure_leg_delta)
    ground = TerrainObservation(cfg.scene.spec_fn).to(device)
    if camera_cfg is not None:
      camera = sim.scene["walk_depth"]
      memory = DepthMemory(n, device, sim.step_dt, camera_cfg.pattern)
      patch = MeasuredPatch(ground.offsets)
      future_feet = FutureFeet(motion)
      from supermimic.tasks.rough_reference import RoughFootReference
      rough_ref = (None if runner.rough_reference in ("off", "nominal") else
                   RoughFootReference(future_feet, runner.rough_reference, ground))
    if rollout_probe is not None:
      rollout_probe.bind(sim, motion, future_feet)
    if action_controller is not None:
      action_controller.bind(sim, motion, runner, patch)
    if base_terrain is not None:
      grid_min, grid_max = ground.offsets.amin(0), ground.offsets.amax(0)
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
    measure_ankles = runner.rough_reference != "off"
    if measure_ankles:
      ankle_correction = torch.zeros_like(xy_error)
      corrected_steps = torch.zeros_like(counts)
      switch_error = torch.zeros_like(xy_error)
      switch_steps = torch.zeros_like(counts)
    history = None
    slip_distance = torch.zeros_like(xy_error)
    stance_time = torch.zeros_like(xy_error)
    measure_slip = (direct_reference or action_controller is not None
                    or (runner.flat_depth_arm is not None and rough_seed is None)
                    or leg_ablation is not None or base_terrain is not None)
    if measure_slip:
      foot_ids, _ = motion.robot.find_sites(["left_foot", "right_foot"])
      if len(foot_ids) != 2:
        raise ValueError("Expected both sole sites for the stance-slip metric")
      previous_feet = previous_stance = None
      landing_history = torch.zeros((5, n), device=device, dtype=torch.long)
      landings = torch.zeros_like(counts)
      failed_landings = torch.zeros_like(counts)
    limit = max(frames) if max_steps is None else min(max_steps, max(frames))
    for t in range(limit):
      if measure_slip:
        previous_stance_at_start = previous_stance
        feet = motion.robot.data.site_pos_w[:, foot_ids]
        near = feet[..., 2] - ground.height_at(feet[..., :2]) < .02
        if previous_feet is not None:
          landed = (near & ~previous_stance & active[:, None]).sum(1)
          landings += landed
          landing_history[t % 5] = landed
          stance = near & previous_stance & active[:, None]
          slip_distance += ((feet[..., :2] - previous_feet[..., :2]).norm(dim=-1) * stance).sum(1)
          stance_time += stance.sum(1) * sim.step_dt
        previous_feet = feet.clone()
        previous_stance = near
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
        if runner.camera_profile == "instinct":
          cnn_image, _ = camera.observe(memory, aux, terrain_only=runner.rough_reference != "off")
        else:
          cnn_image = data.distances
          memory.update(data.distances, data.pos_w, data.quat_w)
        patch.bind(*patch.capture(memory))
        aux = representation.extract_aux_state(obs["full_state"])
        if camera_diagnostic is not None:
          camera_diagnostic.record(sim, camera, motion, future_feet, ground, patch, aux, active)
        td["depth_tokens"] = patch.tokens(aux)
        if rough_ref is None:
          td["future_feet"] = future_feet(motion.time_steps)
        else:
          td["future_feet"], ankle_delta, _ = rough_ref.targets(motion.time_steps, aux, patch)
        if measure_ankles:
          ankles = future_feet.body_indices
          target_z = motion.body_pos_relative_w[:, ankles, 2]
          if rough_ref is None:
            ankle_delta = torch.zeros_like(target_z)
          error = (target_z + ankle_delta - motion.robot_body_pos_w[:, ankles, 2]).abs().mean(-1)
          nominal = future_feet.xyz[motion.time_steps, :, 2]
          near_switch = (nominal[:, 0] - nominal[:, 1]).abs() < 0.04
          on_switch = active & near_switch
          ankle_correction += ankle_delta.abs().mean(-1) * active
          corrected_steps += active & ankle_delta.ne(0).any(-1)
          switch_error += error * on_switch
          switch_steps += on_switch
        base_samples = patch.sample(aux, patch.base_offsets)
        heights = base_samples[..., 0]
        td["terrain"] = torch.zeros_like(heights) if base_terrain == "off" else heights
        if base_terrain is not None:
          visible = base_samples[..., 1].sum(-1).long()
          now = motion.time_steps
          ref = motion.motion
          ends = ref.motion_ends[torch.searchsorted(ref.motion_ends, now, right=True)]
          lookahead = torch.minimum(now + 12, ends - 1)
          yaw = yaw_quat(ref.root_quat_w[now])
          travel = quat_apply_inverse(yaw, ref.root_pos_w[lookahead] - ref.root_pos_w[now])
          future_yaw = yaw_quat(ref.root_quat_w[lookahead])
          turn = (yaw * future_yaw).sum(-1).abs() < 0.99145  # 15 degrees in 12 steps
          x, y = travel[:, :2].unbind(-1)
          group = torch.full_like(now, 4)
          group = torch.where(x > 0.05, 0, group)
          group = torch.where(x < -0.05, 3, group)
          group = torch.where((y.abs() > x.abs()) & (y.abs() > 0.05), 1, group)
          group = torch.where(turn, 2, group)
          nominal = td["future_feet"][..., :2]
          inside = ((nominal >= grid_min) & (nominal <= grid_max)).all(-1).sum((1, 2))
          steps_by_group = (group[:, None] == group_ids) & active[:, None]
          terrain_counts += steps_by_group[..., None] * torch.stack(
            (torch.ones_like(visible), visible, inside, torch.full_like(inside, 6)), -1
          )[:, None]
          if t:
            touchdown = near & ~previous_stance_at_start & active[:, None]
            for slot, horizon in enumerate((4, 8, 12)):
              if t < horizon:
                break
              old_root = root_history[(t - horizon) % 12]
              old_group = group_history[(t - horizon) % 12]
              old_yaw = yaw_quat(old_root[:, 3:7])[:, None].expand(-1, 2, -1).contiguous()
              local = quat_apply_inverse(old_yaw, feet - old_root[:, None, :3])[..., :2]
              covered = ((local >= grid_min) & (local <= grid_max)).all(-1)
              tally = torch.stack(((touchdown & covered).sum(1), touchdown.sum(1)), -1)
              landing_counts[:, :, slot] += (old_group[:, None] == group_ids)[..., None] * tally[:, None]
          root_history[t % 12] = aux
          group_history[t % 12] = group
        if vision is not None:
          td["camera_latent"] = vision.encode(cnn_image)
      elif runner.terrain_inputs != "blind":
        td["terrain"] = (ground.features(representation.extract_aux_state(obs["full_state"]))
                         if rough_seed is not None else rep.new_zeros(n, TerrainObservation.num_features))
      if runner.use_root_waypoints:
        td["root_waypoints"] = root_waypoints(obs["goal"])
      if rollout_probe is not None:
        rollout_probe.record(t, active)
        rollout_probe.adapt(td, representation, motion.time_steps, aux)
      action = (actor(td) if action_controller is None else
                action_controller.act(actor, td, representation, motion.time_steps, aux, active, t))
      action = torch.where(active[:, None], action, 0.0)
      obs, reward, terminated, truncated, _ = sim.step(action)
      ended = sim.termination_manager.get_term("motion_end")
      done = terminated | truncated
      if rollout_probe is not None:
        rollout_probe.record_end(t + 1, active, terminated)
      if camera_cfg is not None:
        memory.reset(done)
      for name, mask in reasons.items():
        mask |= active & done & sim.termination_manager.get_term(name)
      if (active & truncated & ~ended & ~terminated).any():
        raise RuntimeError("Unexpected timeout during a flat regression clip")
      totals["return"] += torch.where(active, reward, 0.0)
      counts += active.long()
      if measure_slip:
        failed_landings += (active & terminated) * landing_history.sum(0)
      completed |= active & ended & ~terminated
      active &= ~done
      if t % 1000 == 0:
        print(f"Frame {t}, {active.sum().item()} remaining", flush=True)
      if not active.any():
        break
    if active.any() and limit == max(frames):
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
        if leg_ablation is not None:
          row.update(landings=int(landings[i]), landings_followed_by_failure_5steps=int(failed_landings[i]),
                     mean_abs_applied_leg_delta=float(correction_sum[i] / counts[i]))
    if base_terrain is not None:
      for row, measurements, touchdowns in zip(rows, terrain_counts.tolist(), landing_counts.tolist(), strict=True):
        row["terrain_by_motion"] = {
          group: dict(steps=step_count, valid_points=valid, nominal_inside=inside, nominal_total=total,
                      touchdown_coverage=dict(zip(("4", "8", "12"), coverage, strict=True)))
          for group, (step_count, valid, inside, total), coverage
          in zip(motion_groups, measurements, touchdowns, strict=True)
        }
    if measure_ankles:
      for i, row in enumerate(rows):
        row.update(
          mean_abs_ankle_correction_m=float(ankle_correction[i] / counts[i]),
          ankle_correction_step_fraction=float(corrected_steps[i] / counts[i]),
          near_switch_ankle_z_error_m=float(switch_error[i] / switch_steps[i]) if switch_steps[i] else None,
          near_switch_steps=int(switch_steps[i]),
        )
    return dict(
      checkpoint=str(checkpoint),
      actor_sha256=hashlib.sha256((checkpoint / "actor.pt").read_bytes()).hexdigest(),
      depth_input=("instinct_torso_camera" if runner.camera_profile == "instinct" else "noiseless_pelvis_camera") if camera_cfg is not None else None,
      completed=int(completed.sum()),
      motions=n,
      terrain_seed=rough_seed,
      rough_scale=rough_scale if rough_seed is not None else None,
      direct_reference=direct_reference,
      leg_ablation=leg_ablation,
      base_terrain=base_terrain,
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
  p.add_argument("--rough-scale", type=float, default=1.0)
  p.add_argument("--generated-reference-dir", type=Path,
                 help="Evaluate checksum-verified terrain-adapted clips without a second root-height reset")
  p.add_argument("--leg-ablation", choices=("on", "off"))
  p.add_argument("--base-terrain", choices=("on", "off"), help="Measure or zero only the Actor terrain input")
  p.add_argument("--online-student-weights", type=Path, help="Apply a trained student to raw motion clips")
  p.add_argument("--mask-online-depth", action="store_true", help="Disable the new residual using its depth mask")
  p.add_argument("--max-steps", type=int, help="Limit the evaluation horizon for each clip")
  a = p.parse_args()
  if not np.isfinite(a.rough_scale) or a.rough_scale <= 0:
    p.error("rough-scale must be finite and positive")
  if a.max_steps is not None and a.max_steps <= 0:
    p.error("max-steps must be positive")
  if a.mask_online_depth and a.online_student_weights is None:
    p.error("mask-online-depth requires online-student-weights")
  if a.online_student_weights is not None and a.generated_reference_dir is not None:
    p.error("Online evaluation uses raw clips, without generated references")
  protocol = json.loads(a.protocol.read_text())
  if a.generated_reference_dir is not None:
    manifest = json.loads((a.generated_reference_dir / "manifest.json").read_text())
    if len(protocol["clips"]) != 40 or len(manifest["motion_clips"]) != 40:
      raise ValueError("Expected a complete LAFAN40 generated reference set")
    if a.rough_seed is not None or a.rough_scale != 1.0:
      p.error("Generated references set their own terrain seed and scale")
    a.rough_seed = int(manifest["terrain_seed"])
    a.rough_scale = float(manifest["rough_scale"])
    if not np.isfinite(a.rough_scale) or a.rough_scale <= 0:
      raise ValueError("Generated terrain scale must be finite and positive")
    for clip in protocol["clips"]:
      name = Path(clip["path"]).stem
      record = manifest["motion_clips"][name]
      generated = a.generated_reference_dir / f"{name}.npz"
      if (record["source_sha256"] != clip["sha256"] or
          record["frames"] != clip["frames"] or
          hashlib.sha256(generated.read_bytes()).hexdigest() != record["adapted_sha256"]):
        raise ValueError(f"Generated reference does not match the protocol: {name}")
      clip["path"] = str(generated.resolve())
      clip["sha256"] = record["adapted_sha256"]
  controller = None
  if a.online_student_weights is not None:
    from supermimic.tasks.online_terrain_adapter import OnlineTerrainActions

    controller = OnlineTerrainActions(a.online_student_weights, a.checkpoint / "actor.pt", a.device,
                                      mask_depth=a.mask_online_depth)
  result = evaluate(
    a.checkpoint, protocol, a.starts, a.device,
    rough_seed=a.rough_seed, save_starts=a.save_starts,
    rough_scale=a.rough_scale, leg_ablation=a.leg_ablation, base_terrain=a.base_terrain,
    direct_reference=a.generated_reference_dir is not None, max_steps=a.max_steps,
    action_controller=controller,
  )
  if controller is not None:
    with a.online_student_weights.open("rb") as source:
      result["online_student_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    result["online_depth_masked"] = a.mask_online_depth
  a.output.parent.mkdir(exist_ok=True, parents=True)
  a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(result["completed"], result["means"])
