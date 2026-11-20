"""Closed-loop G1 push episodes and clean simulator transitions for object dynamics."""

import json
import math
from pathlib import Path

import mjlab.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply

from supermimic.representations import extract_actor_goal, root_waypoints
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.push_box import PushBox, PushCourse
from supermimic.tasks.push_guidance import PushGuidance, stopping_reference
from supermimic.tasks.task_spec import build_actor_obs_td


class PushRollout:
  """Only resets write robot or box poses. Each subsequent step uses normal G1 actuation."""

  def __init__(
    self,
    checkpoint: Path,
    motions: list[Path],
    n: int,
    mass: float,
    friction: float,
    seed: int,
    device: str,
    center_x: float = 0.76,
    lateral_bias: float = 0.0,
    terrain_height: float = 0.0,
    terrain_phase: int = 0,
    grade: float = 0.0,
  ):
    self.device = device
    self.n = n
    self.mass, self.friction = mass, friction
    self.box_spec = PushBox(
      center=(center_x, lateral_bias, 0.41),
      half_size=(0.34, 0.27, 0.41),
      mass=mass,
      floor_friction=friction,
    )
    cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
    cfg.seed = seed
    cfg.scene.env_spacing = 0.0
    cfg.scene.spec_fn = PushCourse(
      height=terrain_height, phase_cells=terrain_phase, box=self.box_spec, grade=grade
    )
    self.grade = grade
    cfg.sim.mujoco.iterations = cfg.sim.mujoco.ls_iterations = 60
    cfg.sim.njmax = max(cfg.sim.njmax or 0, 512)
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)
    cfg.events = {
      k: v
      for k, v in cfg.events.items()
      if k in ("foot_friction", "base_com", "encoder_bias")
    }
    box = ContactMatch(mode="geom", pattern="push_box_collision")
    for key, pattern in [
      ("hand_box", ("left_hand_collision", "right_hand_collision")),
      ("other_box", ".*_collision"),
    ]:
      exclude = (
        ("left_hand_collision", "right_hand_collision") if key == "other_box" else ()
      )
      cfg.scene.sensors += (
        ContactSensorCfg(
          name=key,
          primary=ContactMatch(
            mode="geom", entity="robot", pattern=pattern, exclude=exclude
          ),
          secondary=box,
          fields=("found",),
          reduce="maxforce",
        ),
      )
    cfg.scene.sensors += (
      ContactSensorCfg(
        name="palm_wrench",
        primary=ContactMatch(
          mode="geom",
          entity="robot",
          pattern=("left_hand_collision", "right_hand_collision"),
        ),
        secondary=box,
        fields=("force",),
        reduce="netforce",
      ),
    )
    command = cfg.commands["motion"]
    command.motion_file = [str(p) for p in motions]
    command.sampling_mode = "start"
    command.pose_range, command.velocity_range, command.joint_position_range = (
      {},
      {},
      (0.0, 0.0),
    )
    self.env = create_env(cfg, n, device)
    self.sim = self.env.unwrapped
    self.command = self.sim.command_manager.get_term("motion")
    self.motion = self.command.motion
    default_cfg = SuperMimicRunnerCfg(
      residual_correction=True, residual_shared_trunk=True
    )
    manifest = json.loads((checkpoint / "run_manifest.json").read_text())
    if manifest.get("runner_cfg"):
      default_cfg = load_runner_cfg(manifest)
    self.actor_terrain = default_cfg.terrain_inputs != "blind"
    self.use_root_waypoints = default_cfg.use_root_waypoints
    self.guidance = None
    self.paths = motions
    self.wm = build_world_model(default_cfg, self.env, device)
    self.wm.load_bundle(
      torch.load(checkpoint / "world_model.pt", map_location=device, weights_only=True)
    )
    self.actor = build_actor(default_cfg, self.wm, device)
    self.actor.load_state_dict(
      torch.load(checkpoint / "actor.pt", map_location=device, weights_only=True)
    )
    self.wm.dynamics.eval().requires_grad_(False)
    self.actor.eval().requires_grad_(False)
    mj = self.sim.sim.mj_model
    joint = mj.joint("push_box_free")
    self.box_q = int(joint.qposadr[0])
    self.box_v = int(joint.dofadr[0])
    self.wrists = [
      self.command.cfg.body_names.index(s + "_wrist_yaw_link")
      for s in ("left", "right")
    ]
    self.ankles = [
      self.command.cfg.body_names.index(s + "_ankle_roll_link")
      for s in ("left", "right")
    ]
    self.site_offsets = torch.as_tensor(
      [mj.site("robot/" + s + "_palm").pos.tolist() for s in ("left", "right")],
      dtype=torch.float32,
      device=device,
    )

  def reset(self, seed, gap, offset, distance, scenario_repeat=1):
    self.sim.reset(seed=seed)
    ends = self.motion.motion_ends.tolist()
    lanes = torch.arange(self.n, device=self.device) % len(ends)
    for lane, start in enumerate([0, *ends[:-1]]):
      ids = torch.where(lanes == lane)[0]
      self.command.reset_to_frame(ids, start)
    generator = torch.Generator(device=self.device).manual_seed(seed)
    if self.n % scenario_repeat:
      raise ValueError("Candidate groups must have equally many scenarios")
    spread = torch.rand(
      (self.n // scenario_repeat, 2), device=self.device, generator=generator
    ).repeat(scenario_repeat, 1)
    q = self.sim.sim.data.qpos[:, self.box_q : self.box_q + 7]
    q.zero_()
    q[:, 0] = self.box_spec.center[0] + (spread[:, 0] - 0.5) * gap
    q[:, 1] = self.box_spec.center[1] + (spread[:, 1] - 0.5) * offset
    q[:, 2] = self.box_spec.half_size[2] + 0.0005
    xy = q[:, :2]
    offsets = q.new_tensor(
      [[-0.34, -0.27], [-0.34, 0.27], [0.0, 0.0], [0.34, -0.27], [0.34, 0.27]]
    )
    q[:, 2] += self.wm.terrain_query.height_at(xy[:, None, :] + offsets).max(-1).values
    q[:, 3] = 1.0
    if self.grade:
      angle = -math.atan(self.grade)
      q[:, 2] = (
        self.grade * q[:, 0]
        + self.box_spec.half_size[2] * math.hypot(1.0, self.grade)
        + 0.0005
      )
      q[:, 3] = math.cos(angle / 2)
      q[:, 5] = math.sin(angle / 2)
    self.sim.sim.data.qvel[:, self.box_v : self.box_v + 6] = 0.0
    self.sim.scene.write_data_to_sim()
    self.sim.sim.forward()
    self.sim.sim.sense()
    mj = self.sim.sim.mj_model
    native = mujoco.MjData(mj)
    box_geom = mj.geom("push_box_collision").id
    min_gaps = []
    for pose in self.sim.sim.data.qpos.cpu().numpy():
      native.qpos[:] = pose
      mujoco.mj_forward(mj, native)
      gaps = [
        contact.dist
        for contact in native.contact
        if box_geom in contact.geom
        and any(mj.geom(int(g)).name.startswith("robot/") for g in contact.geom)
      ]
      min_gaps.append(min([0.0, *gaps]))
    self.start_penetration = torch.from_numpy(-np.array(min_gaps)).to(self.device)
    self.command.update_relative_body_poses()
    obs = self.sim.observation_manager.compute(update_history=True)
    start_xy = q[:, :2].clone()
    goal = start_xy.clone()
    goal[:, 0] += distance
    return obs, start_xy, goal, lanes

  def object_state(self):
    data = self.sim.sim.data
    pose = data.qpos[:, self.box_q : self.box_q + 7]
    twist = data.qvel[:, self.box_v : self.box_v + 6]
    return torch.cat([pose, twist[:, :3], quat_apply(pose[:, 3:7], twist[:, 3:])], -1)

  def palms(self, pos, rot):
    return pos[:, self.wrists] + (
      rot[:, self.wrists] @ self.site_offsets[:, :, None]
    ).squeeze(-1)

  def real_palms(self):
    pos = self.command.robot_body_pos_w[:, self.wrists]
    quat = self.command.robot_body_quat_w[:, self.wrists]
    return pos + quat_apply(quat, self.site_offsets.expand_as(pos))

  @torch.no_grad()
  def run(
    self,
    seed,
    steps=145,
    gap=0.10,
    offset=0.12,
    distance=0.67,
    policy=None,
    explore=0.0,
    record=False,
    feedback=False,
    gains=None,
    scenario_repeat=1,
    force_adaptive=False,
    push_force=0.0,
    stop_at_goal=False,
    video_path=None,
  ):
    obs, start, goal, lanes = self.reset(seed, gap, offset, distance, scenario_repeat)
    ctx = self.wm.sim_context.read(self.sim)
    alive = torch.ones(self.n, dtype=torch.bool, device=self.device)
    hands = torch.zeros((self.n, 2), dtype=torch.bool, device=self.device)
    simultaneous = torch.zeros(self.n, dtype=torch.int32, device=self.device)
    other = torch.zeros_like(alive)
    nonhands = self.sim.scene["other_box"]
    parts = torch.zeros(
      (self.n, len(nonhands.primary_names)), dtype=torch.bool, device=self.device
    )
    hold = torch.zeros_like(alive, dtype=torch.int32)
    terminal_box = self.object_state()
    initial = torch.cat([obs["full_state"], terminal_box, ctx], -1).cpu()
    video_poses = []
    if video_path:
      video_poses.append(self.sim.sim.data.qpos[0].clone())
    height_at = self.wm.terrain_query.height_at
    robot_base_height = height_at(obs["full_state"][:, :2])
    box_base_height = height_at(start)
    robot_climb = torch.zeros_like(box_base_height)
    box_climb = torch.zeros_like(box_base_height)
    samples = []
    generator = torch.Generator(device=self.device).manual_seed(seed + 771)
    if (feedback or push_force) and self.guidance is None:
      self.guidance = PushGuidance(
        self.paths, self.wm.dynamics.action_scale, self.device
      )
    if feedback:
      gains = torch.as_tensor(
        PushGuidance.default_gains if gains is None else gains,
        dtype=torch.float32,
        device=self.device,
      ).expand(self.n, -1)
    phase = self.command.time_steps.float().clone()
    bias = None
    returns = torch.zeros(self.n, device=self.device)
    impulse = torch.zeros_like(returns)
    elapsed = torch.zeros_like(returns)
    measuring = torch.zeros_like(alive)
    blend = torch.zeros_like(returns)
    for t in range(steps):
      if (feedback or stop_at_goal) and t:
        self.command.set_time_steps(slice(None), phase.long())
        self.command.update_relative_body_poses()
        obs = self.sim.observation_manager.compute(update_history=False)
      full = obs["full_state"]
      rep = self.wm.representation.represent(full)
      aux = self.wm.representation.extract_aux_state(full)
      frame = self.command.time_steps.clone()
      box = self.object_state()
      palms = self.real_palms()
      task_goal = obs["goal"]
      pace = torch.ones_like(phase)
      if stop_at_goal:
        task_goal, pace = stopping_reference(
          self.motion, phase, self.motion.motion_ends[lanes] - 1, aux, box, goal
        )
      td = build_actor_obs_td(
        {"motion": extract_actor_goal(task_goal)}, rep.unsqueeze(1)
      )
      if self.actor_terrain:
        td["terrain"] = self.wm.terrain_query.features(aux)
      if self.use_root_waypoints:
        td["root_waypoints"] = root_waypoints(task_goal)
      raw = self.actor(td)
      amplitude = blend[:, None] if force_adaptive else 1.0
      if feedback:
        ankle_q = self.command.robot_body_quat_w[:, self.ankles]
        offset = quat_apply(
          ankle_q, box.new_tensor([0.132, 0.0, -0.025]).expand(self.n, 2, 3)
        )
        toes = self.command.robot_body_pos_w[:, self.ankles] + offset
        vel = self.command.robot_body_lin_vel_w[:, self.ankles] + torch.linalg.cross(
          self.command.robot_body_ang_vel_w[:, self.ankles], offset
        )
        correction, rate = self.guidance.command(
          box, palms, toes, vel, goal, frame, gains
        )
        raw = raw + amplitude * correction
        if force_adaptive:
          rate = 1.0 + blend * (rate - 1.0)
        pace = pace * rate
      if feedback or stop_at_goal:
        phase = torch.minimum(phase + pace, self.motion.motion_ends[lanes] - 1)
      if policy is not None:
        raw = raw + amplitude * policy(rep, aux, box, palms, goal)
      if push_force:
        raw += self.guidance.push_effort(
          box, palms, goal, frame, impulse / elapsed.clamp_min(1.0), push_force
        )
      if explore:
        if t % 12 == 0:
          bias = torch.randn(raw.shape, generator=generator, device=self.device).clamp(
            -2.0, 2.0
          )
        raw = raw + explore * bias
      obs, reward, done, _ = self.env.step(raw)
      live = alive & ~done.bool()
      if video_path and bool(live[0]):
        video_poses.append(self.sim.sim.data.qpos[0].clone())
      contacted = self.sim.scene["hand_box"].data.found > 0
      hands |= live[:, None] & contacted
      simultaneous += (live & contacted.all(-1)).int()
      this_other = nonhands.data.found > 0
      parts |= live[:, None] & this_other
      other |= live & this_other.any(-1)
      after = self.object_state()
      robot_climb = torch.maximum(
        robot_climb,
        torch.where(live, height_at(obs["full_state"][:, :2]) - robot_base_height, 0.0),
      )
      box_climb = torch.maximum(
        box_climb, torch.where(live, height_at(after[:, :2]) - box_base_height, 0.0)
      )
      # Integrate the forward palm impulse over the push, using previous steps only
      # for control. A first collision impulse alone is a poor estimate of resistance.
      wrench = self.sim.scene["palm_wrench"].data.force.sum(1)[:, 0].abs()
      measuring |= live & contacted.any(-1)
      sample = live & measuring & ((goal - box[:, :2])[:, 0] > 0.10)
      elapsed += sample.float()
      impulse += torch.where(sample, wrench.clamp_max(180.0), 0.0)
      mean_force = impulse / elapsed.clamp_min(1.0)
      blend = ((mean_force - 7.0) / 8.0).clamp(0.0, 1.0) * (elapsed / 10.0).clamp_max(
        1.0
      )
      terminal_box = torch.where(live[:, None], after, terminal_box)
      upright = 1 - 2 * (after[:, 4].square() + after[:, 5].square())
      at_goal = (after[:, :2] - goal).abs()
      in_zone = (at_goal[:, 0] < 0.15) & (at_goal[:, 1] < 0.15) & (upright > 0.94)
      hold = torch.where(live & in_zone, hold + 1, 0)
      returns += alive * reward
      if record:
        samples.append(
          dict(
            full=full.clone(),
            after_full=obs["full_state"].clone(),
            action=raw.clone(),
            box=box,
            after_box=after,
            palms=palms,
            after_palms=self.real_palms(),
            contact=contacted,
            other_contact=this_other.any(-1),
            valid=live,
            ctx=ctx,
            frame=frame,
            goal=goal,
          )
        )
      alive = live
    good = (
      alive & (hold >= 10) & hands.all(-1) & ~other & (self.start_penetration < 0.001)
    )
    speed = terminal_box[:, 7:9].norm(dim=-1)
    rows = dict(
      robot_course_climb_m=robot_climb,
      box_course_climb_m=box_climb,
      lane=lanes,
      success=good,
      survived=alive,
      bimanual=hands.all(-1),
      other_contact=other,
      simultaneous_steps=simultaneous,
      terminal_speed_mps=speed,
      parked=good & (speed < 0.1),
      upright=(1 - 2 * (terminal_box[:, 4].square() + terminal_box[:, 5].square()))
      > 0.94,
      dx=terminal_box[:, 0] - start[:, 0],
      lateral_error=(terminal_box[:, 1] - goal[:, 1]).abs(),
      goal_error=(terminal_box[:, :2] - goal).norm(dim=-1),
      hold_steps=hold,
      tracking_return=returns,
      start_penetration_m=self.start_penetration,
      palm_load_n=impulse / elapsed.clamp_min(1.0),
      residual_blend=blend,
    )
    for i, name in enumerate(nonhands.primary_names):
      if parts[:, i].any():
        rows["box_contact_" + name] = parts[:, i]
    if video_path:
      self.write_video(
        video_path, torch.stack(video_poses).cpu().numpy(), goal[0].tolist()
      )
    return (
      {k: v.cpu() for k, v in rows.items()},
      initial,
      (
        {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        if samples
        else None
      ),
    )

  def write_video(self, path, poses, goal):
    """Replay the first simulated episode. The green outline marks the target center zone."""
    import imageio.v2 as imageio

    model = self.sim.sim.mj_model
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth = 115.0
    camera.elevation = -13.0
    camera.distance = 3.05
    camera.lookat[:] = [0.84, 0.0, 0.71]
    x, y = goal
    corners = [
      (x - 0.15, y - 0.15),
      (x + 0.15, y - 0.15),
      (x + 0.15, y + 0.15),
      (x - 0.15, y + 0.15),
    ]
    xy = torch.tensor(corners, device=self.device)
    h = self.wm.terrain_query.height_at(xy).cpu().numpy()
    corners = np.column_stack([corners, h + 0.008])
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
      mujoco.Renderer(model, height=384, width=640) as renderer,
      imageio.get_writer(path, fps=50, codec="libx264", quality=8) as writer,
    ):
      for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        sc = renderer.scene
        for i in range(4):
          marker = sc.geoms[sc.ngeom]
          mujoco.mjv_initGeom(
            marker,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([0.1, 1.0, 0.12, 1.0], np.float32),
          )
          mujoco.mjv_connector(
            marker,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.008,
            corners[i],
            corners[(i + 1) % 4],
          )
          sc.ngeom += 1
        writer.append_data(renderer.render())
