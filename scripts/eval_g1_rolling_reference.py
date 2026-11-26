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


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


class RollingReference:
  """Single-episode planner and paired recorder. Physical terrain is never queried here."""

  horizon = 50
  period = 4

  def __init__(self, clip, enabled):
    self.clip = clip
    self.enabled = enabled
    self.trace = []
    self.plans = []
    self.phase = 0
    self.stop = None
    self.anchors = [None, None]
    self.last = None

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
    self.contacts = [sim.scene[k] for k in CONTACTS]
    order = [str(name).split("/")[-1] for name in self.contacts[0].primary_names]
    self.order = [order.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    self.robot = command.robot
    self.body_ids = self.robot.find_bodies(
      ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
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
    self.actual = (
      self.robot.data.body_link_pos_w[0, self.body_ids].cpu().numpy() - self.origin
    )
    for foot in range(2):
      if not self.contact[foot]:
        self.anchors[foot] = None
      elif self.anchors[foot] is None:
        self.anchors[foot] = self.actual[foot].copy()

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

  def replan(self):
    start = self.phase
    end = len(self.raw) - 1
    ids = np.minimum(start + np.arange(self.horizon), end)
    mode, z, valid = self.query(ids)
    stance = self.support[ids]
    touchdowns = np.argwhere(stance[1:] & ~stance[:-1])
    landing = None
    if len(touchdowns):
      offset, foot = map(int, touchdowns[0])
      landing = (offset + 1, foot)
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
        support_xy = self.xyz[ids[at], 1 - foot, :2]
        target_xy = self.xyz[ids[at], foot, :2]
        direction = support_xy - target_xy
        distance = np.linalg.norm(direction)
        lengths = np.minimum([0.03, 0.06, 0.09], 0.35 * distance)
        offsets = lengths[:, None] * direction / max(distance, 1e-6)
        candidates = np.zeros((len(offsets), 2, 2))
        candidates[:, foot] = offsets
        choices, heights, hits = self.query(
          np.repeat(ids[at], len(offsets)), candidates
        )
        samples, mask = heights[:, foot], hits[:, foot]
        span = np.where(mask, samples, -np.inf).max(1) - np.where(
          mask, samples, np.inf
        ).min(1)
        known = np.flatnonzero((choices[:, foot] > 0) & (span < 0.025))
        if len(known):
          delta = offsets[known[0]]
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
    root_dz = (lift * stance).sum(1) / stance.sum(1).clip(1)
    poses = src.copy()
    poses[:, 2] += gaussian_filter1d(root_dz, 2, mode="nearest").clip(-0.045, 0.045)
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

  def adapt(self, obs, representation, motion_times, aux):
    self.camera.observe(self.memory, aux, terrain_only=True)
    if self.enabled and self.step % self.period == 0:
      before = perf_counter()
      self.replan()
      self.plans[-1]["planning_ms"] = (perf_counter() - before) * 1000
    local = self.step % self.period
    if self.enabled:
      at = torch.full_like(motion_times, local)
      obs["goal_motion"] = representation.compute_wm_goal(self.ref, at, aux)[1]
      obs["future_feet"] = self.feet(at)
    planned = self.cache[local] if self.enabled else self.xyz[self.step]
    self.trace.append(
      dict(
        step=self.step,
        phase=self.phase,
        nominal=self.xyz[self.phase].tolist(),
        reference=planned.tolist(),
        four_step_reference=(
          self.cache[local + 4] if self.enabled else self.xyz[self.step + 4]
        ).tolist(),
        actual=self.actual.tolist(),
        contact=self.contact.tolist(),
      )
    )
    if not self.enabled or self.stop is None or self.phase < self.stop:
      self.phase = min(self.phase + 1, len(self.raw) - 1)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument(
    "--motion", type=Path, required=True, help="Raw walk2_subject1 LAFAN G1 clip"
  )
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--steps", type=int, default=100)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  if args.motion.stem != "walk2_subject1":
    p.error("scenario frame indices refer to walk2_subject1")
  manifest = json.loads((args.checkpoint / "run_manifest.json").read_text())
  if (
    manifest["runner_cfg"]["imagined_vision"] != "latent"
    or manifest["max_iterations"] != 30000
  ):
    p.error("expected the 30K latent checkpoint")
  if args.steps < 1:
    p.error("--steps must be positive")
  args.output.mkdir(parents=True, exist_ok=False)
  results = {}
  metadata = dict(
    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
    actor_sha256=digest(args.checkpoint / "actor.pt"),
    source_sha256=digest(args.motion),
    scene_seed=1715,
    rough_scale=0.45,
    steps=args.steps,
  )
  for name, first in (("forward", 9121), ("return", 9518), ("backward", 9587)):
    folder = args.output / name
    folder.mkdir()
    path = folder / args.motion.name
    with np.load(args.motion) as src:
      length = args.steps + RollingReference.horizon + 2
      if first + length > len(src["joint_pos"]):
        raise ValueError("Requested horizon extends past the source clip")
      clip = {
        k: (v[first : first + length].copy() if k not in ("joint_names", "fps") else v)
        for k, v in src.items()
      }
    clip["root_pos_w"][:, :2] -= clip["root_pos_w"][0, :2].copy()
    np.savez(path, **clip)
    protocol = dict(
      seed=42,
      clips=[dict(path=str(path), frames=length, local_start=0, sha256=digest(path))],
    )
    results[name] = {}
    for arm in ("raw", "online"):
      torch.manual_seed(42)
      probe = RollingReference(path, arm == "online")
      row = evaluate(
        args.checkpoint,
        protocol,
        folder / "starts.npz",
        args.device,
        rough_seed=1715,
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
      if len(actual) > 4:
        issued = np.array([x["four_step_reference"] for x in probe.trace[:-4]])
        results[name][arm]["ankle_z_mae_4step_cm"] = float(
          np.abs(actual[4:, ..., 2] - issued[..., 2]).mean() * 100
        )
      print(name, arm, results[name][arm], flush=True)
  (args.output / "summary.json").write_text(
    json.dumps(dict(metadata=metadata, scenes=results), indent=2)
  )


if __name__ == "__main__":
  main()
