"""Build a short, straight G1 walking reference with both palms on a box's rear face."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def select_start(clip, length, distance, steady=False, push_feasible=False):
  pos = clip["root_pos_w"][:, :2]
  orientation = Rotation.from_quat(clip["root_quat_w"], scalar_first=True)
  facing = orientation.as_matrix()[:, :2, 0]
  if push_feasible:
    model = get_spec().compile()
    data = mujoco.MjData(model)
    cols = [model.joint(str(name)).qposadr[0] for name in clip["joint_names"]]
    ids = [model.geom(side + "_foot4_collision").id for side in ("left", "right")]
    toe_end = np.empty((len(pos), 2, 2, 2))
    for i in range(len(pos)):
      data.qpos[:3], data.qpos[3:7], data.qpos[cols] = (
        clip["root_pos_w"][i],
        clip["root_quat_w"][i],
        clip["joint_pos"][i],
      )
      mujoco.mj_kinematics(model, data)
      vector = (
        data.geom_xmat[ids, :].reshape(2, 3, 3)[:, :2, 2]
        * model.geom_size[ids, 1, None]
      )
      relative = data.geom_xpos[ids, :2] - pos[i]
      toe_end[i] = np.stack([relative - vector, relative + vector], axis=1)
  candidates = []
  durations = (
    [length]
    if not push_feasible
    else [length - 20, length, length + 20, length + 40, length + 65]
  )
  for duration in durations:
    for t in range(0, len(pos) - duration, 5):
      path = pos[t : t + duration] - pos[t]
      advance = np.linalg.norm(path[-1])
      if not 0.45 < advance < 1.25 or (
        push_feasible and abs(advance - distance) > 0.08
      ):
        continue
      forward = path[-1] / advance
      lateral = np.array([-forward[1], forward[0]])
      if (
        np.min(facing[t : t + duration] @ forward) < 0.8
        or abs(path @ lateral).max() > 0.18
        or np.diff(path @ forward).min() < -0.012
      ):
        continue
      stopped = np.linalg.norm(path[-1] - path[-20])
      cost = abs(advance - distance) + 2 * stopped + abs(path @ lateral).max()
      if steady:
        speed = np.linalg.norm(
          clip["root_lin_vel_w"][t + duration - 8 : t + duration, :2], axis=1
        ).mean()
        cost += 0.8 * speed + 0.035 * np.linalg.norm(
          clip["joint_vel"][t + duration - 1, :12]
        )
      if push_feasible:
        # Reject strides whose leading toe can enter the box ahead of a reachable hand.
        reach = (toe_end[t : t + duration] @ forward).max() + model.geom_size[
          ids, 0
        ].max()
        if reach > 0.415:
          continue
        cost += 0.4 * abs(reach - 0.32)
      candidates.append((cost + 0.002 * abs(duration - length), t, duration))
  if not candidates:
    raise ValueError("No straight walk toward a stop in the source clip")
  return min(candidates)[1:]


def solve_palm_target(model, data, ids, palm, previous, target, nominal, limit_change):
  q, v = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
  jac = np.empty((3, model.nv))

  def residual(x):
    data.qpos[q] = x
    mujoco.mj_kinematics(model, data)
    return np.r_[
      data.site_xpos[palm] - target, 0.012 * (x - previous), 0.008 * (x - nominal)
    ]

  def derivative(x):
    data.qpos[q] = x
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacSite(model, data, jac, None, palm)
    return np.vstack([jac[:, v], 0.012 * np.eye(len(q)), 0.008 * np.eye(len(q))])

  lo, hi = model.jnt_range[ids].T
  if limit_change:
    lo = np.maximum(lo, previous - 0.055)
    hi = np.minimum(hi, previous + 0.055)
  return least_squares(
    residual, np.clip(previous, lo, hi), jac=derivative, bounds=(lo, hi), max_nfev=24
  ).x


def prepare(
  source,
  dest,
  length=125,
  distance=0.85,
  clearance=False,
  steady=False,
  push_feasible=False,
  hand_height=0.785,
  hand_lead=0.0,
):
  with np.load(source) as archive:
    clip = dict(archive)
  t, duration = select_start(clip, length, distance, steady, push_feasible)
  n = len(clip["joint_pos"])
  clip = {
    k: (v[t : t + duration].copy() if v.ndim and len(v) == n else v.copy())
    for k, v in clip.items()
  }
  if duration != length:
    times = np.arange(duration)
    phase = np.linspace(0, duration - 1, length)
    for key in (
      "root_pos_w",
      "root_lin_vel_w",
      "root_ang_vel_w",
      "joint_pos",
      "joint_vel",
    ):
      values = clip[key]
      resampled = np.stack([np.interp(phase, times, v) for v in values.T], axis=-1)
      if "vel" in key:
        resampled *= (duration - 1) / (length - 1)
      clip[key] = resampled.astype(values.dtype)
    ori = Slerp(times, Rotation.from_quat(clip["root_quat_w"], scalar_first=True))(
      phase
    )
    clip["root_quat_w"] = ori.as_quat(scalar_first=True).astype(
      clip["root_quat_w"].dtype
    )
  root = clip["root_pos_w"]
  delta = root[-1, :2] - root[0, :2]
  heading = Rotation.from_euler("z", -np.arctan2(delta[1], delta[0]))
  root[:, :2] -= root[0, :2].copy()
  root[:] = heading.apply(root)
  clip["root_quat_w"][:] = (
    heading * Rotation.from_quat(clip["root_quat_w"], scalar_first=True)
  ).as_quat(scalar_first=True)
  for key in ("root_lin_vel_w", "root_ang_vel_w"):
    clip[key][:] = heading.apply(clip[key])
  model = get_spec().compile()
  data = mujoco.MjData(model)
  columns = np.array(
    [model.joint(str(name)).qposadr[0] for name in clip["joint_names"]]
  )
  qj = clip["joint_pos"]
  fps = float(clip["fps"].item())
  reach = np.full(length, 0.30)
  if clearance:
    waist = np.flatnonzero(clip["joint_names"] == "waist_pitch_joint").item()
    qj[:, waist] += 0.21 if steady else 0.27
    toes = [model.geom(s + "_foot4_collision").id for s in ("left", "right")]
    for i in range(length):
      data.qpos[:3], data.qpos[3:7], data.qpos[columns] = (
        root[i],
        clip["root_quat_w"][i],
        qj[i],
      )
      mujoco.mj_kinematics(model, data)
      tip = (
        data.geom_xpos[toes, 0]
        + abs(data.geom_xmat[toes, 2]) * model.geom_size[toes, 1]
        + model.geom_size[toes, 0]
      )
      reach[i] = (
        np.clip(tip.max() - root[i, 0] + 0.10, 0.37, 0.40)
        if steady
        else np.clip(tip.max() - root[i, 0] + 0.10, 0.385, 0.445)
      )
  errors = []
  for side, sign in (("left", 1.0), ("right", -1.0)):
    ids = [
      j
      for j in range(model.njnt)
      if model.joint(j).name.startswith(side + "_")
      and any(part in model.joint(j).name for part in ("shoulder", "elbow", "wrist"))
    ]
    q, v = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    arm = np.array([np.flatnonzero(columns == col).item() for col in q])
    palm = model.site(side + "_palm").id
    prev = np.array([-0.65, sign * 0.12, 0.0, 0.65, 0.0, 0.0, 0.0])
    jac = np.empty((3, model.nv))
    for i in range(length):
      data.qpos[:3], data.qpos[3:7], data.qpos[columns] = (
        root[i],
        clip["root_quat_w"][i],
        qj[i],
      )
      data.qpos[q] = prev
      target = np.array([root[i, 0] + reach[i] + hand_lead, sign * 0.165, hand_height])
      if steady:
        target[1] += 0.65 * root[i, 1]
        nominal = np.array([-0.85, sign * 0.12, 0.0, 0.65, 0.0, 0.0, 0.0])
        previous = prev.copy()

        prev = solve_palm_target(
          model, data, ids, palm, previous, target, nominal, i > 0
        )
        data.qpos[q] = prev
        mujoco.mj_kinematics(model, data)
        errors.append(float(np.linalg.norm(data.site_xpos[palm] - target)))
        qj[i, arm] = prev
        continue
      for _ in range(30):
        mujoco.mj_kinematics(model, data)
        error = target - data.site_xpos[palm]
        if np.linalg.norm(error) < 0.0006:
          break
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jac, None, palm)
        j = jac[:, v]
        update = j.T @ np.linalg.solve(j @ j.T + 1.0e-4 * np.eye(3), error)
        data.qpos[q] = np.clip(
          data.qpos[q] + np.clip(update, -0.12, 0.12), *model.jnt_range[ids].T
        )
      mujoco.mj_kinematics(model, data)
      errors.append(float(np.linalg.norm(target - data.site_xpos[palm])))
      prev = data.qpos[q].copy()
      qj[i, arm] = prev
    clip["joint_vel"][:, arm] = np.gradient(qj[:, arm], axis=0) * fps
  # The actor can look ahead 45 frames. Keep the final standing target well-defined.
  for key in (
    "joint_pos",
    "joint_vel",
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
  ):
    value = clip[key]
    tail = np.repeat(value[-1:], 70, axis=0)
    if "vel" in key:
      tail.fill(0.0)
    clip[key] = np.concatenate([value, tail])
  dest.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(dest, **clip)
  return dict(
    source=source.name,
    first_frame=t,
    source_frames=duration,
    walking_frames=length,
    reference_travel_m=float(root[-1, 0]),
    palm_ik_p95_m=float(np.quantile(errors, 0.95)),
    palm_ik_max_m=max(errors),
  )


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("source", type=Path)
  p.add_argument("dest", type=Path)
  p.add_argument("--length", type=int, default=125)
  p.add_argument("--distance", type=float, default=0.85)
  p.add_argument("--clearance", action="store_true")
  p.add_argument("--steady", action="store_true")
  p.add_argument("--push-feasible", action="store_true")
  p.add_argument("--hand-height", type=float, default=0.785)
  p.add_argument("--hand-lead", type=float, default=0.0)
  args = p.parse_args()
  print(
    json.dumps(
      prepare(
        args.source,
        args.dest,
        length=args.length,
        distance=args.distance,
        clearance=args.clearance,
        steady=args.steady,
        push_feasible=args.push_feasible,
        hand_height=args.hand_height,
        hand_lead=args.hand_lead,
      ),
      indent=2,
    )
  )
