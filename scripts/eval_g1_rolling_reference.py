"""Paired frozen-policy G1 replay with a causal, depth-conditioned reference window."""

import argparse
import hashlib
import json
import subprocess
from copy import copy
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch
from eval_scene_flat_regression import evaluate
from probe_g1_ankle_optimization import geometry
from probe_g1_terrain_reference import poses_from_clip
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from train_g1_online_terrain_ppo import CONTACTS, contact_config

from supermimic.tasks.depth_mimic import DepthMemory, walking_camera
from supermimic.tasks.terrain_reference import TerrainLegReference

SCENES = {"forward": 9121, "return": 9518, "backward": 9587, "slow_backward": 11650}


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


class RollingReference:
  """Single-episode planner and paired recorder. Physical terrain is never queried here."""

  horizon = 50
  period = 4

  def __init__(self, clip, enabled, support_brake=False, blind_steps=0):
    self.blind_steps = blind_steps
    self.clip = clip
    self.enabled = enabled
    self.support_brake = support_brake
    self.brake_target = None
    self.release_count = 0
    self.contact_ticks = np.zeros(2, dtype=int)
    self.trace = []
    self.plans = []
    self.phase = 0
    self.stop = None
    self.anchors = [None, None]
    self.pivots = [None, None]
    self.last = None
    self.use_window = enabled

  def configure(self, cfg):
    cfg.scene.sensors = (
      *cfg.scene.sensors,
      contact_config(CONTACTS[0], "walking_surface"),
      contact_config(CONTACTS[1], "terrain"),
    )

  def bind(self, sim, command, feet):
    self.sim = sim
    self.camera = sim.scene["walk_depth"]
    self.memory = DepthMemory(
      1, sim.device, sim.step_dt, walking_camera("instinct").pattern
    )
    if self.support_brake:
      # The simulator supplies local odometry on static terrain. Retain identity-
      # tagged measured cells through a walking step and a subsequent reversal.
      self.memory.max_age = round(2.0 / sim.step_dt)
      self.memory.reset(torch.ones(1, device=sim.device, dtype=torch.bool))
    self.contacts = [sim.scene[k] for k in CONTACTS]
    order = [str(name).split("/")[-1] for name in self.contacts[0].primary_names]
    self.order = [order.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    self.robot = command.robot
    self.body_ids = self.robot.find_bodies(
      ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    self.sole_ids = self.robot.find_sites(
      ["left_foot", "right_foot"], preserve_order=True
    )[0]
    model = sim.sim.mj_model
    self.ik = TerrainLegReference(model, None)
    self.feet = copy(feet)
    self.ref = copy(command.motion)
    self.feet.motion = self.ref
    self.feet._feet_table = None
    self.device = sim.device
    with np.load(self.clip) as clip:
      self.raw = poses_from_clip(clip, np.arange(len(clip["joint_pos"])), model)
    self.address = [
      model.jnt_qposadr[model.joint("robot/" + n).id] for n in self.ref.fk.joint_names
    ]
    self.xyz, self.rot, soles = geometry(self.ik, self.raw)
    # Preserve the mocap ankle arc about the planted sole's lowest material point.
    # The ankle itself moves during heel strike and toe off.
    contact_points = soles.reshape(len(soles), 2, -1, 3)
    self.clearance = contact_points[..., 2].min(-1)
    lowest = contact_points[..., 2].argmin(axis=2)
    world_points = np.take_along_axis(
      contact_points, lowest[..., None, None], axis=2
    ).squeeze(2)
    self.sole_offsets = np.einsum("tfji,tfj->tfi", self.rot, world_points - self.xyz)

    self.support = (soles[..., 2].min(axis=(-1, -2)) < 0.035) & (
      np.linalg.norm(np.gradient(self.xyz, axis=0), axis=-1) / sim.step_dt < 0.45
    )
    self.origin = sim.scene.env_origins[0].cpu().numpy()
    self.contact = np.zeros(2, bool)
    self.cache = None

  def record(self, step, active):
    self.step = step
    self.contact = (
      ((self.contacts[0].data.found > 0) | (self.contacts[1].data.found > 0))[
        0, self.order
      ]
      .cpu()
      .numpy()
      .astype(bool)
    )
    self.pelvis = self.robot.data.root_link_pos_w[0].cpu().numpy() - self.origin
    self.contact_ticks = np.where(self.contact, self.contact_ticks + 1, 0)
    self.velocity = self.robot.data.root_link_lin_vel_w[0].cpu().numpy()
    self.actual = (
      self.robot.data.body_link_pos_w[0, self.body_ids].cpu().numpy() - self.origin
    )
    for foot in range(2):
      if not self.contact[foot]:
        self.anchors[foot] = None
      elif self.anchors[foot] is None:
        self.anchors[foot] = self.actual[foot].copy()
        if self.support_brake:
          lever = self.sole_offsets[self.phase, foot]
          quat = self.robot.data.body_link_quat_w[0, self.body_ids[foot]].cpu().numpy()
          pivot = self.actual[foot] + Rotation.from_quat(quat, scalar_first=True).apply(
            lever
          )
          # Root tracking error can exceed the leg correction bound. Anchor in
          # the nominal footprint neighborhood rather than twisting the stance
          # leg to absorb the full horizontal odometry-to-reference discrepancy.
          reference = self.xyz[self.phase, foot] + self.rot[self.phase, foot] @ lever
          delta = pivot[:2] - reference[:2]
          future = min(self.phase + 10, len(self.raw) - 1)
          travel = self.raw[future, :2] - self.raw[self.phase, :2]
          speed = np.linalg.norm(travel) / (10 * self.sim.step_dt)
          # Faster strides provide a shorter stance interval for correction.
          limit = np.clip(0.045 / max(speed, 0.25), 0.025, 0.12)
          pivot[:2] = reference[:2] + delta * min(
            1.0, limit / max(np.linalg.norm(delta), 1e-8)
          )
          self.pivots[foot] = pivot, lever

  def record_end(self, step, active, terminated):
    pass

  def query(self, frames, shift=None):
    # Five taps aligned with nominal foot axes, in the same world as the camera points.
    taps = np.array(
      [[0, 0, 0], [0.08, 0, 0], [-0.08, 0, 0], [0, 0.04, 0], [0, -0.04, 0]]
    )
    points = self.xyz[frames, :, None] + np.einsum(
      "tfij,pj->tfpi", self.rot[frames], taps
    )
    if shift is not None:
      points[..., :2] += shift[:, :, None, :]
    xy = torch.tensor(
      (points + self.origin)[None, ..., :2], device=self.device, dtype=torch.float32
    )
    fresh_z, fresh = self.memory.at_world(xy, max_age=0)
    recent_z, recent = self.memory.at_world(xy)
    fresh, recent = fresh[0].cpu().numpy(), recent[0].cpu().numpy()

    # Require a center measurement and coverage on both axes for a usable foothold.
    def covered(hits):
      return hits[..., 0] & hits[..., 1:3].any(-1) & hits[..., 3:5].any(-1)

    mode = np.where(covered(fresh), 2, np.where(covered(recent), 1, 0))
    z = (
      np.where(fresh, fresh_z[0].cpu().numpy(), recent_z[0].cpu().numpy())
      - self.origin[2]
    )
    return mode, z, recent

  def shorter_landing(self, frame, foot):
    support_xy = self.xyz[frame, 1 - foot, :2]
    target_xy = self.xyz[frame, foot, :2]
    direction = support_xy - target_xy
    distance = np.linalg.norm(direction)
    lengths = np.minimum([0.03, 0.06, 0.09], 0.35 * distance)
    offsets = lengths[:, None] * direction / max(distance, 1e-6)
    if self.support_brake:
      # Also search the swing foot's actual approach line. Side-to-side
      # shortening alone misses previously observed cells in front of a retreat.
      approach = self.actual[foot, :2] - target_xy
      travel = np.linalg.norm(approach)
      along = approach / max(travel, 1e-6)
      across = np.array([-along[1], along[0]])
      offsets = np.vstack(
        (
          offsets,
          [
            length * along + lateral * across
            for length in np.minimum([0.035, 0.07, 0.11, 0.15], 0.5 * travel)
            for lateral in (0.0, -0.035, 0.035, -0.065, 0.065)
          ],
        )
      )
      offsets = offsets[np.argsort(np.linalg.norm(offsets, axis=1), kind="stable")]
    candidates = np.zeros((len(offsets), 2, 2))
    candidates[:, foot] = offsets
    choices, heights, hits = self.query(np.repeat(frame, len(offsets)), candidates)
    samples, mask = heights[:, foot], hits[:, foot]
    span = np.where(mask, samples, -np.inf).max(1) - np.where(
      mask, samples, np.inf
    ).min(1)
    viable = (choices[:, foot] > 0) & (span < 0.025)
    if self.support_brake:
      # Keep the swing foot on its own side and within its nominal reach.
      left_axis = Rotation.from_quat(
        self.raw[frame, 3:7], scalar_first=True
      ).as_matrix()[:2, 1]
      signed_width = (
        (target_xy + offsets - support_xy) @ left_axis * (1 if foot == 0 else -1)
      )
      viable &= signed_width >= 0.09
    known = np.flatnonzero(viable)
    return offsets[known[0]] if len(known) else None

  def apply_support_reference(self, poses, orientation, target):
    # Keep both measured ankles planted. Advance the pelvis smoothly from the
    # measured moving state toward the midpoint of the support segment.
    if self.brake_target is None:
      self.release_count = 0
      self.brake_target = np.mean(self.anchors, axis=0)[:2]
      self.brake_step = self.step
      self.brake_velocity = self.velocity[:2].copy()
      self.brake_pose = self.sim.sim.data.qpos[0].cpu().numpy().copy()
      self.brake_pose[:3] -= self.origin
      self.brake_feet = np.stack(self.anchors)
      _, rotations, _ = geometry(self.ik, self.brake_pose[None])
      self.brake_orientation = rotations[0]
    # Starting IK from an airborne mocap pose can leave a planted foot above
    # the ground at the joint correction bound. Seed from measured support.
    elapsed = (self.step - self.brake_step + np.arange(len(poses))) * self.sim.step_dt
    duration = 0.40
    u = np.minimum(elapsed / duration, 1.0)[:, None]
    blend = u * u * (3.0 - 2.0 * u)
    poses[:] = self.brake_pose
    # Keep planted joints reachable and limit the trunk attitude change.
    # Only the roll/pitch from the moving capture are eased toward upright.
    rot = Rotation.from_quat(self.brake_pose[3:7], scalar_first=True)
    ypr = rot.as_euler("ZYX")
    angles = np.tile(ypr, (len(poses), 1))
    angles[:, 1:] -= blend * np.clip(ypr[1:], -0.12, 0.12)
    poses[:, 3:7] = Rotation.from_euler("ZYX", angles).as_quat(scalar_first=True)

    orientation[:] = self.brake_orientation
    # A new physical touchdown defines a new support anchor after a recovery step.
    for side, anchor in enumerate(self.anchors):
      if anchor is not None and np.linalg.norm(anchor - self.brake_feet[side]) > 0.035:
        self.brake_feet[side] = anchor
    center = self.brake_feet.mean(0)[:2]
    # Preserve the sagittal ankle-to-pelvis offset (the ankle sits behind
    # the pressure center), without moving beyond a small support polygon.
    center += np.clip(self.brake_pose[:2] - self.brake_target, -0.045, 0.045)
    poses[:, :2] = (
      (1.0 - blend) * self.brake_pose[:2]
      + blend * center
      + duration * u * (1.0 - u) ** 2 * self.brake_velocity
    )
    target[:] = self.brake_feet

  def replan(self):
    start = self.phase
    end = len(self.raw) - 1
    ids = np.minimum(start + np.arange(self.horizon), end)
    mode, z, valid = self.query(ids)
    stance = self.support[ids]
    touchdowns = np.argwhere(stance[1:] & ~stance[:-1])
    landing = None
    # Once the expected foot has physically landed at that location, occlusion
    # beneath the sole must not hold the phase at the same touchdown forever.
    for offset, foot in touchdowns.tolist():
      at = offset + 1
      already_landed = (
        self.support_brake
        and at <= self.period
        and self.contact_ticks[foot] >= 3
        and np.linalg.norm(self.actual[foot, :2] - self.xyz[ids[at], foot, :2]) < 0.16
      )
      if not already_landed:
        landing = (at, foot)
        break
    stop = None
    shift = np.zeros((len(ids), 2, 2))
    decision = "no_touchdown_in_window"
    observed = None
    if landing is not None:
      at, foot = landing
      observed = int(mode[at, foot])
      decision = ("unknown", "remembered", "visible")[observed]
      if observed == 0:
        # Shorten along the nominal step toward the other (support) foot, within
        # the small-perturbation range of leg IK. Check all candidate sole areas.
        delta = self.shorter_landing(ids[at], foot)
        if delta is not None:
          # Keep the correction through stance and taper during the next swing.
          liftoff = np.flatnonzero(~stance[at:, foot])
          release = at + int(liftoff[0]) if len(liftoff) else len(ids)
          blend = np.minimum(np.arange(len(ids)) / max(at, 1), 1.0)
          blend *= np.clip(1 - (np.arange(len(ids)) - release) / 8.0, 0, 1)
          shift[:, foot] = (blend * blend * (3 - 2 * blend))[:, None] * delta
          mode, z, valid = self.query(ids, shift)
          decision = "shorten"
        else:
          # Finish an airborne step continuously and brake in double support.
          doubles = np.flatnonzero(stance.all(1))
          if self.contact.all():
            stop = start
          elif len(doubles):
            stop = start + int(doubles[0])
          decision = (
            "seek_support"
            if stop is None
            else "hold"
            if stop == start
            else "finish_to_support"
          )
    if self.support_brake and self.brake_target is not None:
      # A dropped contact sample must not resume the moving mocap reference.
      # Confirm a usable landing on two consecutive maps before releasing.
      self.release_count = (
        self.release_count + 1
        if decision in ("shorten", "visible", "remembered", "no_touchdown_in_window")
        else 0
      )
      if self.release_count < 2:
        stop = start
        shift.fill(0)
        decision = "brake_hold"
    if stop is not None:
      ids = np.minimum(ids, stop)
      mode, z, valid = self.query(ids)
      stance = self.support[ids]
    self.stop = stop
    src = self.raw[ids]
    target = self.xyz[ids].copy()
    target[..., :2] += shift
    floor = np.where(valid, z, -np.inf).max(-1)
    lift = np.where(mode > 0, np.clip(floor, -0.05, 0.07), 0.0)
    if self.support_brake:
      # Mid-swing already has clearance. Taper the landing height correction as
      # nominal clearance rises, keeping a 12 mm measured-obstacle margin.
      gap = self.clearance[ids]
      near = np.clip(1.0 - np.maximum(gap, 0.0) / 0.045, 0.0, 1.0)
      required = np.maximum(lift + 0.012 - gap, 0.0)
      swing = np.maximum(lift * near, np.minimum(lift, required))
      lift = np.where(stance, lift, swing)
    # Unknown swing samples keep nominal clearance; no zero-height observation enters IK.
    target[..., 2] += lift
    orientation = self.rot[ids].copy()
    for foot in range(2):
      # Estimate shallow roll/pitch from opposing sole taps only when both are observed.
      for axis, a, b, span, sign in ((1, 1, 2, 0.16, -1), (0, 3, 4, 0.08, 1)):
        slope = np.where(
          valid[:, foot, a] & valid[:, foot, b] & (mode[:, foot] > 0),
          np.arctan2(z[:, foot, a] - z[:, foot, b], span),
          0.0,
        )
        vec = np.zeros((len(ids), 3))
        vec[:, axis] = sign * slope.clip(-0.12, 0.12)
        orientation[:, foot] = (
          orientation[:, foot] @ Rotation.from_rotvec(vec).as_matrix()
        )
      if self.anchors[foot] is not None and stance[0, foot]:
        planted = np.logical_and.accumulate(stance[:, foot])
        target[planted, foot] = self.anchors[foot]
        if self.support_brake:
          # A constant ankle in world coordinates overconstrains toe-off. Keep
          # the pivot fixed and let the ankle follow the reference orientation.
          pivot, lever = self.pivots[foot]
          target[planted, foot] = pivot - orientation[planted, foot] @ lever
    root_dz = (lift * stance).sum(1) / stance.sum(1).clip(1)
    poses = src.copy()
    poses[:, 2] += gaussian_filter1d(root_dz, 2, mode="nearest").clip(-0.045, 0.045)
    if (
      self.support_brake
      and stop == start
      and (self.contact.all() or self.brake_target is not None)
    ):
      self.apply_support_reference(poses, orientation, target)
      src = poses.copy()
    elif stop != start:
      self.brake_target = None
    # Bounded damped least squares on the two six-DoF legs. Keep upper body and root XY.
    m, d = self.ik.model, self.ik.data
    for i, pose in enumerate(poses):
      for _ in range(2):
        d.qpos[:] = pose
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        for side, ankle in enumerate(self.ik.ankles):
          mujoco.mj_jacBody(m, d, self.ik.jpos, self.ik.jrot, ankle)
          jac = np.vstack([self.ik.jpos, 0.16 * self.ik.jrot])[:, self.ik.daddr[side]]
          rot_err = Rotation.from_matrix(
            orientation[i, side] @ d.xmat[ankle].reshape(3, 3).T
          ).as_rotvec()
          err = np.r_[target[i, side] - d.xpos[ankle], 0.16 * rot_err]
          addr = self.ik.qaddr[side]
          delta = np.linalg.solve(jac.T @ jac + 0.002 * np.eye(6), jac.T @ err)
          lo, hi = self.ik.limits[side].T
          pose[addr] = np.clip(
            pose[addr] + delta,
            np.maximum(np.minimum(lo, src[i, addr]), src[i, addr] - 0.24),
            np.minimum(np.maximum(hi, src[i, addr]), src[i, addr] + 0.24),
          )
    # Blend overlapping predictions for the same wall-clock instant, including phase braking.
    if self.last is not None:
      overlap = self.period
      previous = self.last[self.period : 2 * self.period]
      alpha = np.linspace(0, 1, overlap)[:, None]
      cols = np.r_[0:3, self.address]
      poses[:overlap, cols] = (
        previous[:, cols] * (1 - alpha) + poses[:overlap, cols] * alpha
      )
      # Normalized shortest-arc blending avoids a yaw jump on pause/resume.
      old_q = previous[:, 3:7]
      new_q = poses[:overlap, 3:7]
      new_q = np.where((old_q * new_q).sum(1, keepdims=True) < 0, -new_q, new_q)
      quat = old_q * (1 - alpha) + new_q * alpha
      poses[:overlap, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    self.last = poses.copy()
    self.set_window(poses)
    self.plans.append(
      dict(
        step=self.step,
        phase=start,
        landing=landing,
        mode=observed,
        decision=decision,
        stop=stop,
        max_shortening_m=float(np.linalg.norm(shift, axis=-1).max()),
      )
    )

  def set_window(self, poses):
    def tensor(a):
      return torch.as_tensor(a, device=self.device, dtype=torch.float32)

    self.ref.root_pos_w = tensor(poses[:, :3])
    self.ref.root_quat_w = tensor(poses[:, 3:7])
    self.ref.joint_pos = tensor(poses[:, self.address])
    self.ref.joint_vel = tensor(
      np.gradient(poses[:, self.address], self.sim.step_dt, axis=0)
    )
    self.ref.root_lin_vel_w = tensor(
      np.gradient(poses[:, :3], self.sim.step_dt, axis=0)
    )
    self.ref.motion_ends = torch.tensor([len(poses)], device=self.device)
    xyz, rot, _ = geometry(self.ik, poses)
    self.feet.xyz = tensor(xyz)
    self.feet.quat = tensor(
      Rotation.from_matrix(rot.reshape(-1, 3, 3))
      .as_quat(scalar_first=True)
      .reshape(len(poses), 2, 4)
    )
    self.cache = xyz

  def observe_map(self, aux):
    self.camera.observe(self.memory, aux, terrain_only=True)

  def adapt(self, obs, representation, motion_times, aux):
    if self.step >= self.blind_steps:
      self.observe_map(aux)
    if self.enabled and self.step % self.period == 0:
      before = perf_counter()
      self.replan()
      self.plans[-1]["planning_ms"] = (perf_counter() - before) * 1000
    local = self.step % self.period
    if self.use_window:
      at = torch.full_like(motion_times, local)
      obs["goal_motion"] = representation.compute_wm_goal(self.ref, at, aux)[1]
      obs["future_feet"] = self.feet(at)
    planned = self.cache[local] if self.use_window else self.xyz[self.step]
    self.trace.append(
      dict(
        step=self.step,
        phase=self.phase,
        nominal=self.xyz[self.phase].tolist(),
        reference=planned.tolist(),
        four_step_reference=(
          self.cache[local + 4]
          if self.use_window
          else self.xyz[min(self.step + 4, len(self.xyz) - 1)]
        ).tolist(),
        actual=self.actual.tolist(),
        pelvis=self.pelvis.tolist(),
        pelvis_velocity=self.velocity.tolist(),
        contact=self.contact.tolist(),
        sole_xy_speed=self.robot.data.site_lin_vel_w[0, self.sole_ids, :2]
        .norm(dim=-1)
        .cpu()
        .tolist(),
      )
    )
    if not self.enabled or self.stop is None or self.phase < self.stop:
      self.phase = min(self.phase + 1, len(self.raw) - 1)


class ProprioceptiveDepthMemory(DepthMemory):
  """Exclude likely self returns using only link poses and measured range points."""

  def update_cloud(self, cloud, valid, pos, quat):
    # Conservative body envelopes may discard ground close to a foot. Those
    # cells remain unknown; simulated ray-hit labels are never consulted.
    near_link = torch.cdist(cloud, self.links).amin(dim=-1) < 0.13
    super().update_cloud(cloud, valid & self.fresh[:, None] & ~near_link, pos, quat)


class PhaseLockedReference(RollingReference):
  """Correct the raw reference locally in height while keeping its clock and XY."""

  def bind(self, sim, command, feet):
    super().bind(sim, command, feet)
    self.memory = ProprioceptiveDepthMemory(
      1, sim.device, sim.step_dt, walking_camera("instinct").pattern
    )
    self.original = command.motion
    # Small vertical corrections around each nominal leg pose. Precompute the
    # damped task Jacobian response using only the supplied motion and G1 model.
    self.response = np.empty((len(self.raw), 2, 6))
    model, data = self.ik.model, self.ik.data
    for i, pose in enumerate(self.raw):
      data.qpos[:] = pose
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      for side, ankle in enumerate(self.ik.ankles):
        mujoco.mj_jacBody(model, data, self.ik.jpos, self.ik.jrot, ankle)
        J = np.vstack((self.ik.jpos, 0.16 * self.ik.jrot))[:, self.ik.daddr[side]]
        self.response[i, side] = np.linalg.solve(J.T @ J + 0.002 * np.eye(6), J[2])

  def observe_map(self, aux):
    self.memory.links = self.robot.data.body_link_pos_w
    self.memory.fresh = self.camera.refresh_mask
    self.camera.observe(self.memory, aux, terrain_only=False)

  def replan(self):
    ids = np.minimum(self.phase + np.arange(self.horizon), len(self.raw) - 1)
    mode, z, valid = self.query(ids)
    floor = np.where(valid, z, -np.inf).max(-1)
    height = np.where(mode > 0, np.clip(floor, -0.05, 0.07), 0.0)
    # A uniformly elevated support surface translates the body and both feet
    # together. Only a measured height difference should flex the legs.
    gap = np.maximum(self.clearance[ids], 0.0)
    weight = (mode > 0) * np.exp(-gap / 0.035)
    count = gaussian_filter1d(weight.sum(1), 3, mode="nearest")
    weighted = gaussian_filter1d((weight * height).sum(1), 3, mode="nearest")
    root = np.divide(weighted, np.maximum(count, 0.25)).clip(-0.045, 0.045)
    relative = height - root[:, None]
    near = np.clip(1.0 - gap / 0.045, 0.0, 1.0)
    dz = np.maximum(
      relative * near, np.minimum(relative, np.maximum(0.0, relative + 0.012 - gap))
    )
    dz = gaussian_filter1d(np.where(mode > 0, dz, 0.0), 1.5, axis=0, mode="nearest")
    poses = self.raw[ids].copy()
    poses[:, 2] += root
    joint = self.ik.qaddr
    dq = self.response[ids] * dz[..., None]
    base = poses[:, joint]
    poses[:, joint] = np.clip(
      base + dq.clip(-0.18, 0.18),
      np.minimum(base, self.ik.limits[..., 0]),
      np.maximum(base, self.ik.limits[..., 1]),
    )
    if self.last is not None:
      overlap = self.period
      alpha = np.linspace(0, 1, overlap)[:, None]
      poses[:overlap] = (
        self.last[overlap : 2 * overlap] * (1 - alpha) + poses[:overlap] * alpha
      )
    self.last = poses.copy()
    self.set_window(poses)
    # Preserve the recorded velocities under an identity correction, including
    # window edges, and add only the derivative of the changed pose.
    frames = torch.as_tensor(ids, device=self.device)
    delta = poses - self.raw[ids]
    self.use_window = bool(np.max(np.abs(delta)) > 1e-8)
    vel = np.gradient(delta, self.sim.step_dt, axis=0)
    self.ref.joint_vel = self.original.joint_vel[frames] + torch.as_tensor(
      vel[:, self.address], device=self.device, dtype=torch.float32
    )
    self.ref.root_lin_vel_w = self.original.root_lin_vel_w[frames] + torch.as_tensor(
      vel[:, :3], device=self.device, dtype=torch.float32
    )
    self.plans.append(
      dict(
        step=self.step,
        phase=self.phase,
        current=int((mode == 2).sum()),
        remembered=int((mode == 1).sum()),
        unknown=int((mode == 0).sum()),
        peak_root_m=float(abs(delta[:, 2]).max()),
        peak_joint_rad=float(abs(delta[:, joint]).max()),
      )
    )


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--motion", type=Path, required=True, help="Raw LAFAN G1 clip")
  p.add_argument("--output", type=Path, required=True)
  p.add_argument(
    "--fixed-phase-pair",
    action="store_true",
    help="Pair raw with depth-updated references at the original phase",
  )
  p.add_argument(
    "--full-motion",
    action="store_true",
    help="Pair both arms from frame zero through the episode boundary",
  )
  p.add_argument("--steps", type=int)
  p.add_argument(
    "--planner-blind-steps",
    type=int,
    default=0,
    help="Suppress only planner camera updates at startup to test reacquisition",
  )
  p.add_argument(
    "--scenes",
    nargs="+",
    choices=SCENES,
    default=("forward", "return", "backward"),
  )
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--terrain-seed", type=int, default=1715)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  if args.full_motion:
    if args.steps is not None:
      p.error("--full-motion uses the original clip length")
    args.fixed_phase_pair = True
    with np.load(args.motion) as src:
      args.steps = len(src["joint_pos"])
  elif args.steps is None:
    args.steps = 100
  if not args.full_motion and args.motion.stem != "walk2_subject1":
    p.error("scenario frame indices refer to walk2_subject1")
  manifest = json.loads((args.checkpoint / "run_manifest.json").read_text())
  if (
    manifest["runner_cfg"]["imagined_vision"] != "latent"
    or manifest["max_iterations"] != 30000
  ):
    p.error("expected the 30K latent checkpoint")
  if not 0 <= args.planner_blind_steps <= args.steps:
    p.error("blind steps must be between zero and the evaluation horizon")
  if args.steps < 1:
    p.error("--steps must be positive")
  args.output.mkdir(parents=True, exist_ok=False)
  results = {}
  metadata = dict(
    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
    actor_sha256=digest(args.checkpoint / "actor.pt"),
    source_sha256=digest(args.motion),
    scene_seed=args.terrain_seed,
    seed=args.seed,
    planner_blind_steps=args.planner_blind_steps,
    rough_scale=0.45,
    steps=args.steps,
    fixed_phase_pair=args.fixed_phase_pair,
    full_motion=args.full_motion,
  )
  scenes = {args.motion.stem: 0} if args.full_motion else SCENES
  for name, first in scenes.items():
    if not args.full_motion and name not in args.scenes:
      continue
    folder = args.output / name
    folder.mkdir()
    path = folder / args.motion.name
    with np.load(args.motion) as src:
      length = (
        args.steps if args.full_motion else args.steps + RollingReference.horizon + 2
      )
      if first + length > len(src["joint_pos"]):
        raise ValueError("Requested horizon extends past the source clip")
      clip = {
        k: (v[first : first + length].copy() if k not in ("joint_names", "fps") else v)
        for k, v in src.items()
      }
    if not args.full_motion:
      clip["root_pos_w"][:, :2] -= clip["root_pos_w"][0, :2].copy()
    np.savez(path, **clip)
    protocol = dict(
      seed=args.seed,
      clips=[dict(path=str(path), frames=length, local_start=0, sha256=digest(path))],
    )
    results[name] = {}
    arms = (
      ("raw", "fixed_phase") if args.fixed_phase_pair else ("raw", "online", "support")
    )
    for arm in arms:
      torch.manual_seed(args.seed)
      planner = PhaseLockedReference if arm == "fixed_phase" else RollingReference
      probe = planner(
        path,
        arm != "raw",
        support_brake=arm == "support",
        blind_steps=args.planner_blind_steps,
      )
      row = evaluate(
        args.checkpoint,
        protocol,
        folder / "starts.npz",
        args.device,
        rough_seed=args.terrain_seed,
        rough_scale=0.45,
        save_starts=arm == "raw",
        max_steps=args.steps,
        rollout_probe=probe,
      )
      result = dict(**row, source_start=first, plans=probe.plans, trace=probe.trace)
      (folder / f"{arm}.json").write_text(json.dumps(result, allow_nan=False))
      actual = np.array([x["actual"] for x in probe.trace])
      reference = np.array([x["reference"] for x in probe.trace])
      error = (actual - reference) * 100
      results[name][arm] = dict(
        **row["per_motion"][0],
        last_phase=probe.trace[-1]["phase"],
        ankle_z_mae_cm=float(np.abs(error[..., 2]).mean()),
        ankle_xyz_mean_distance_cm=float(np.linalg.norm(error, axis=-1).mean()),
      )
      held = np.r_[False, np.diff([x["phase"] for x in probe.trace]) == 0]
      contacts = np.array([x["contact"] for x in probe.trace])
      pelvis_speed = np.linalg.norm(
        [x["pelvis_velocity"][:2] for x in probe.trace], axis=1
      )
      sole_speed = np.array([x["sole_xy_speed"] for x in probe.trace])
      results[name][arm].update(
        held_steps=int(held.sum()),
        terminal_pelvis_speed_400ms=float(pelvis_speed[-20:].mean()),
        terminal_double_support_fraction_400ms=float(contacts[-20:].all(1).mean()),
        double_support_held_steps=int((held & contacts.all(1)).sum()),
        mean_held_pelvis_speed=float(pelvis_speed[held].mean()) if held.any() else None,
        mean_contact_sole_speed=float(sole_speed[contacts].mean())
        if contacts.any()
        else None,
      )
      if len(actual) > 4:
        issued = np.array([x["four_step_reference"] for x in probe.trace[:-4]])
        results[name][arm]["ankle_z_mae_4step_cm"] = float(
          np.abs(actual[4:, ..., 2] - issued[..., 2]).mean() * 100
        )
      # Score every arm against the same raw XY at the same wall-clock frames.
      xy = np.array([x["pelvis"][:2] for x in probe.trace])
      desired = clip["root_pos_w"][: args.steps, :2]
      checkpoints = np.unique(np.r_[np.arange(24, args.steps, 25), args.steps - 1])
      reached = [
        i < len(xy) and np.linalg.norm(xy[i] - desired[i]) <= 0.15 for i in checkpoints
      ]
      results[name][arm].update(
        waypoints_reached=int(sum(reached)),
        waypoints=len(checkpoints),
        final_xy_arrived=bool(reached[-1]),
        actual_xy_displacement_m=(xy[-1] - xy[0]).tolist(),
        raw_xy_mae_m=float(np.linalg.norm(xy - desired[: len(xy)], axis=1).mean()),
      )
      if args.full_motion:
        if not results[name][arm]["termination_terms"]:
          raise RuntimeError("Full motion ended without an episode boundary")
        results[name][arm]["final_phase_fraction"] = probe.trace[-1]["phase"] / (
          length - 1
        )
      print(name, arm, results[name][arm], flush=True)
  (args.output / "summary.json").write_text(
    json.dumps(dict(metadata=metadata, scenes=results), indent=2)
  )


if __name__ == "__main__":
  main()
