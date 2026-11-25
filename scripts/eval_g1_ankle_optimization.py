"""Paired oracle checkpoint rollouts over the nine offline retargeting windows."""

import argparse
from copy import copy
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg

from eval_scene_flat_regression import evaluate


class AnkleRollout:
  def __init__(self, clips, trajectories=None):
    self.clips = clips
    self.trajectories = trajectories
    self.trace = [[] for _ in clips]
    self.termination = [None for _ in clips]

  def configure(self, cfg):
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
      name="probe_foot_ground",
      primary=ContactMatch("subtree", ("left_ankle_roll_link", "right_ankle_roll_link"), entity="robot"),
      secondary=ContactMatch("body", "world"),
      fields=("found",), reduce="mindist", num_slots=1,
    ))

  def bind(self, sim, command, feet):
    self.robot = command.robot
    self.contact = sim.scene["probe_foot_ground"]
    self.torso = self.robot.find_bodies("torso_link")[0][0]
    self.dt = sim.step_dt
    if self.trajectories is None:
      return
    original = command.motion
    self.motion = copy(original)
    for field in ("root_pos_w", "root_lin_vel_w", "joint_pos", "joint_vel"):
      setattr(self.motion, field, getattr(original, field).clone())
    self.nominal_feet = feet
    self.feet = copy(feet)
    self.feet.motion = self.motion
    self.feet.xyz, self.feet.quat = feet.xyz.clone(), feet.quat.clone()
    self.feet._feet_table = None
    self.weights = original.root_pos_w.new_zeros(original.time_step_total)
    model = sim.sim.mj_model
    addresses = [model.jnt_qposadr[model.joint(f"robot/{name}").id] for name in original.fk.joint_names]
    for clip, sequence, end in zip(self.clips, self.trajectories, original.motion_ends.tolist(), strict=True):
      begin = end - clip["frames"] + clip["optimization_start"]
      raw, adapted = (sequence[key] for key in ("original", "clearance_optimized"))
      indices = slice(begin, begin + len(raw))
      # Ease into and out of the correction over four control steps.
      ramp = np.minimum(np.arange(len(raw)), np.arange(len(raw))[::-1]) / 4
      blend = 0.5 - 0.5 * np.cos(np.pi * np.clip(ramp, 0, 1))
      delta = (adapted - raw) * blend[:, None]
      root_delta = self.weights.new_tensor(delta[:, :3])
      dq = self.weights.new_tensor(delta[:, addresses])
      root_velocity_delta = torch.gradient(root_delta, spacing=self.dt, dim=0)[0]
      dq_dt = torch.gradient(dq, spacing=self.dt, dim=0)[0]
      self.motion.root_pos_w[indices] += root_delta
      self.motion.root_lin_vel_w[indices] += root_velocity_delta
      self.motion.joint_pos[indices] += dq
      self.motion.joint_vel[indices] += dq_dt
      self.weights[indices] = self.weights.new_tensor(blend)
      frames = torch.arange(begin, begin + len(raw), device=self.weights.device)
      pos, quat, *_ = self.motion.body_state(frames)
      self.feet.xyz[indices] = pos[:, feet.body_indices]
      self.feet.quat[indices] = quat[:, feet.body_indices]

  def adapt(self, obs, representation, motion_times, aux):
    if self.trajectories is None:
      return
    obs["goal_motion"] = representation.compute_wm_goal(self.motion, motion_times, aux)[1]
    ends = self.motion.motion_ends[torch.searchsorted(self.motion.motion_ends, motion_times, right=True)] - 1
    future = torch.minimum(motion_times[:, None] + self.feet.ahead, ends[:, None])
    # The IK result is already tapered to the original reference. Fade the oracle
    # target with the same envelope so the two arms share goals outside the window.
    nominal = self.nominal_feet(motion_times)
    oracle_offset = obs["future_feet"] - nominal
    obs["future_feet"] = self.feet(motion_times) + (1 - self.weights[future][:, None, :, None]) * oracle_offset

  def record(self, step, active):
    contact = (self.contact.data.found > 0).cpu().tolist()
    vertical = self.robot.data.root_link_lin_vel_w[:, 2].cpu().tolist()
    torso = self.robot.data.body_link_quat_w[:, self.torso]
    tilt = torch.rad2deg((1 - 2 * torso[:, 1:3].square().sum(-1)).clamp(-1, 1).acos()).cpu().tolist()
    for i, enabled in enumerate(active.cpu().tolist()):
      if enabled and step < self.clips[i]["horizon"]:
        self.trace[i].append(dict(step=step, contact=contact[i], pelvis_vz_m_s=vertical[i], torso_tilt_deg=tilt[i]))

  def record_end(self, step, active, terminated):
    for i, failed in enumerate((active & terminated).cpu().tolist()):
      if failed and step <= self.clips[i]["horizon"]:
        self.termination[i] = step

  def events(self, index):
    trace = self.trace[index]
    center = self.clips[index]["center_step"]
    return [(n, foot) for n in range(2, len(trace)) for foot in range(2)
            if abs(n - center) <= 10 and trace[n]["contact"][foot]
            and not trace[n - 1]["contact"][foot] and not trace[n - 2]["contact"][foot]]


def contact_comparison(clips, baseline, optimized):
  rows = []
  for i, clip in enumerate(clips):
    center = clip["center_step"]
    pairs = [(a, b, f) for a, f in baseline.events(i) for b, g in optimized.events(i) if f == g]
    pair = min(pairs, key=lambda item: abs(item[0] - center) + abs(item[1] - center)) if pairs else None
    row = {"motion": Path(clip["path"]).stem, "ankle_only_termination_step": baseline.termination[i],
           "optimized_termination_step": optimized.termination[i],
           "contact_sensor_order": list(baseline.contact.primary_names),
           "ankle_only_events": baseline.events(i), "optimized_events": optimized.events(i),
           "ankle_only_trace": baseline.trace[i], "optimized_trace": optimized.trace[i]}
    if pair:
      a, b, foot = pair
      length = min(11, len(baseline.trace[i]) - a, len(optimized.trace[i]) - b)
      row.update(contact_foot=foot, matched_post_contact_samples=length)
      for probe, at, label in ((baseline, a, "ankle_only"), (optimized, b, "optimized")):
        post = probe.trace[i][at:at + length]
        row[label] = dict(contact_step=at, peak_abs_pelvis_vz_m_s=max(abs(x["pelvis_vz_m_s"]) for x in post),
                          peak_torso_tilt_deg=max(x["torso_tilt_deg"] for x in post),
                          terminated_within_200ms=probe.termination[i] is not None and at <= probe.termination[i] <= at + 10)
    rows.append(row)
  return rows


def load_windows(offline, protocol_file):
  report = json.loads((offline / "report.json").read_text())
  protocol = json.loads(protocol_file.read_text())
  clips, trajectories = [], []
  for entry in protocol["clips"]:
    name = Path(entry["path"]).stem
    if name not in report["records"]:
      continue
    record = report["records"][name]
    center = record["window_center"]
    # Recover the exact clipped window start used by the offline probe.
    first = max(0, record["first_failed_step"] - 30)
    optimization_start = max(first, center - 10)
    warmup = max(0, optimization_start - 12)
    clips.append({**entry, "local_start": warmup, "optimization_start": optimization_start,
                  "center_step": center - warmup, "horizon": center - warmup + 22})
    with np.load(offline / f"{name}.npz") as sequence:
      trajectories.append(dict(sequence))
  if len(clips) != 9:
    raise ValueError("Expected the nine paired disagreement clips")
  protocol["clips"] = clips
  return protocol, clips, trajectories, report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("checkpoint", "offline", "protocol", "output"):
    parser.add_argument(name, type=Path)
  args = parser.parse_args()
  protocol, clips, trajectories, report = load_windows(args.offline, args.protocol)
  args.output.mkdir(parents=True, exist_ok=True)
  starts = args.output / "starts.npz"
  baseline, optimized = AnkleRollout(clips), AnkleRollout(clips, trajectories)
  for probe, label in ((baseline, "ankle_only"), (optimized, "optimized")):
    result = evaluate(args.checkpoint, protocol, starts, "cuda:0",
                      rough_seed=report["terrain_seed"], rough_scale=report["rough_scale"],
                      max_steps=max(clip["horizon"] for clip in clips), rollout_probe=probe,
                      save_starts=label == "ankle_only" and not starts.exists())
    (args.output / f"{label}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  rows = contact_comparison(clips, baseline, optimized)
  output = {"step_seconds": baseline.dt, "per_motion": rows}
  (args.output / "comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
  matched = [row for row in rows if "contact_foot" in row]
  print("Matched touchdowns:", len(matched))
  for label in ("ankle_only", "optimized"):
    print(label, "terminations", sum(row[f"{label}_termination_step"] is not None for row in rows))
    if matched:
      for k in ("peak_abs_pelvis_vz_m_s", "peak_torso_tilt_deg", "terminated_within_200ms"):
        print(k, mean(row[label][k] for row in matched))


if __name__ == "__main__":
  main()
