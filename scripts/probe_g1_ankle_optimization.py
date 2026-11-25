"""Compare vertical-only ankle targets and constrained lower-body IK near paired failures."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from probe_g1_terrain_reference import Clearance, collision_scene, poses_from_clip
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from supermimic.tasks.rough_reference import RoughFootReference
from supermimic.tasks.terrain_reference import TerrainLegReference


def geometry(adapter, poses):
  ankles, rotations, soles = [], [], []
  for pose in poses:
    adapter.data.qpos[:] = pose
    mujoco.mj_kinematics(adapter.model, adapter.data)
    ankles.append(adapter.data.xpos[adapter.ankles].copy())
    rotations.append(adapter.data.xmat[adapter.ankles].reshape(2, 3, 3).copy())
    soles.append(adapter.sole_points())
  return tuple(map(np.stack, (ankles, rotations, soles)))


def solve_window(
  adapter, poses, target, rotations, support, *, root_xy=False, root_z=True
):
  """Fit foot pose with joint bounds and temporal regularization on corrections."""
  model, data = adapter.model, adapter.data
  root = (
    np.array([0, 1, 2])
    if root_xy
    else np.array([2])
    if root_z
    else np.array([], dtype=int)
  )
  addr = np.r_[root, adapter.qaddr.ravel()]
  dofs = np.r_[root, adapter.daddr.ravel()]
  count, width = len(poses), len(addr)
  nominal = poses[:, addr]
  limit_lo, limit_hi = adapter.limits.reshape(-1, 2).T
  root_width = len(root)
  root_limit = (
    np.array([0.025, 0.025, 0.06])
    if root_xy
    else np.array([0.06])
    if root_z
    else np.array([])
  )
  lower = np.c_[
    np.tile(-root_limit, (count, 1)),
    np.minimum(limit_lo - nominal[:, root_width:], 0.0),
  ]
  upper = np.c_[
    np.tile(root_limit, (count, 1)), np.maximum(limit_hi - nominal[:, root_width:], 0.0)
  ]
  # Retain pre-existing violations, but permit only movement toward joint limits.
  lower[:, root_width:] = np.maximum(lower[:, root_width:], -0.25)
  upper[:, root_width:] = np.minimum(upper[:, root_width:], 0.25)
  root_penalty = (
    np.array([2.0, 2.0, 0.6])
    if root_xy
    else np.array([0.6])
    if root_z
    else np.array([])
  )
  penalty = sparse.diags(
    np.tile(np.r_[root_penalty, np.full(width - root_width, 0.035)], count)
  )
  diff = sparse.diags(
    [-np.ones(count - 1), np.ones(count - 1)], [0, 1], shape=(count - 1, count)
  )
  smooth = sparse.kron(
    diff, sparse.diags(np.r_[np.ones(root_width), np.full(width - root_width, 0.09)])
  )
  corrected = poses.copy()
  pos_jac = np.zeros((3, model.nv))
  rot_jac = np.zeros_like(pos_jac)
  for _ in range(3):
    jacobians, errors = [], []
    for i, pose in enumerate(corrected):
      data.qpos[:] = pose
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      blocks, residuals = [], []
      for side, ankle in enumerate(adapter.ankles):
        mujoco.mj_jacBody(model, data, pos_jac, rot_jac, ankle)
        dp = target[i, side] - data.xpos[ankle]
        dr = Rotation.from_matrix(
          rotations[i, side] @ data.xmat[ankle].reshape(3, 3).T
        ).as_rotvec()
        weight = (
          np.array([2.0, 2.0, 3.0, 0.16, 0.16, 0.16])
          if support[i, side]
          else np.array([1.0, 1.0, 2.0, 0.10, 0.10, 0.10])
        )
        blocks.append(weight[:, None] * np.vstack((pos_jac[:, dofs], rot_jac[:, dofs])))
        residuals.append(weight * np.r_[dp, dr])
      jacobians.append(np.vstack(blocks))
      errors.append(np.concatenate(residuals))
    task = sparse.block_diag(jacobians, format="csr")
    current = (corrected[:, addr] - nominal).ravel()
    system = sparse.vstack((task, penalty, smooth), format="csr")
    desired = np.r_[
      np.concatenate(errors) + task @ current,
      np.zeros(penalty.shape[0] + smooth.shape[0]),
    ]
    # Strict bounds are required by lsq_linear. Equal bounds are fixed to numerical tolerance.
    fit = lsq_linear(
      system,
      desired,
      bounds=(lower.ravel(), np.maximum(upper, lower + 1e-10).ravel()),
      lsq_solver="lsmr",
      tol=1e-6,
      lsmr_tol=1e-7,
    )
    if not fit.success or not np.isfinite(fit.x).all():
      raise RuntimeError(f"Leg retargeting failed: {fit.message}")
    corrected[:, addr] = nominal + fit.x.reshape(count, width)
  return corrected


def compare_motion(adapter, meter, data, failure_frame, radius):
  # Select the last nominal touchdown near the first early termination.
  first = max(0, failure_frame - 3 * radius)
  end = min(len(data["joint_pos"]), failure_frame + 2 * radius + 13)
  poses = poses_from_clip(data, np.arange(first, end), adapter.model)
  ankles, rotations, soles = geometry(adapter, poses)
  stance = soles[..., 2].min(axis=(-1, -2)) <= 0.02
  contacts = np.flatnonzero((stance[1:] & ~stance[:-1]).any(1)) + 1
  prior = contacts[contacts <= failure_frame - first]
  center = int(prior[-1]) if len(prior) else min(failure_frame - first, len(poses) - 1)
  start, stop = max(0, center - radius), min(len(poses), center + radius + 1)
  frames = torch.arange(start, stop)
  root = torch.as_tensor(poses[:, :7], dtype=torch.float32)

  class OracleFeet(SimpleNamespace):
    def __call__(self, time):
      # The probe consumes the current ankle offset, with the three future slots unused.
      return self.xyz.new_zeros(len(time), 2, 3, 9)

  feet = OracleFeet(
    ahead=torch.tensor([4, 8, 12]),
    xyz=torch.from_numpy(ankles).float(),
    body_indices=adapter.ankles,
    motion=SimpleNamespace(
      root_pos_w=root[:, :3],
      root_quat_w=root[:, 3:],
      motion_ends=torch.tensor([len(root)]),
    ),
  )
  reference = RoughFootReference(
    feet, "oracle", SimpleNamespace(height_at=adapter.height_at)
  )
  _, lift, _ = reference.targets(frames, root[frames], None)
  lift = lift.numpy()
  window = poses[start:stop]
  target = ankles[start:stop].copy()
  target[..., 2] += lift
  optimized = solve_window(
    adapter, window, target, rotations[start:stop], stance[start:stop]
  )
  solved, _, _ = geometry(adapter, optimized)
  raw = ankles[start:stop]
  sole_window = soles[start:stop]
  floor = adapter.height_at(
    torch.as_tensor(sole_window[..., :2], dtype=torch.float32)
  ).numpy()
  clearance = (sole_window[..., 2] - floor).min(axis=(-2, -1))
  # Add just enough lift to resolve the predicted geometry penetration. Keep the
  # original correction where it already clears the surface.
  clearance_lift = np.maximum(lift, -clearance)
  clearance_target = raw.copy()
  clearance_target[..., 2] += clearance_lift
  clearance_pose = solve_window(
    adapter, window, clearance_target, rotations[start:stop], stance[start:stop]
  )
  clearance_ankle, _, clearance_soles = geometry(adapter, clearance_pose)
  clearance_floor = adapter.height_at(
    torch.as_tensor(clearance_soles[..., :2], dtype=torch.float32)
  ).numpy()
  final_clearance = (clearance_soles[..., 2] - clearance_floor).min(axis=(-2, -1))
  support = stance[start:stop]
  protected = np.ones(adapter.model.nq, dtype=bool)
  protected[np.r_[2, adapter.qaddr.ravel()]] = False
  if not np.array_equal(clearance_pose[:, protected], window[:, protected]):
    raise RuntimeError("The retargeter changed a protected coordinate")
  native_depth = np.array(
    [
      [meter.contact(q).max() for q in sequence]
      for sequence in (window, optimized, clearance_pose)
    ]
  )
  clearance_delta = (
    clearance_pose[:, adapter.qaddr.ravel()] - window[:, adapter.qaddr.ravel()]
  )
  delta = optimized[:, adapter.qaddr.ravel()] - window[:, adapter.qaddr.ravel()]
  return {
    "first_failed_step": failure_frame,
    "window_center": first + center,
    "center_is_nominal_touchdown": bool(len(prior)),
    "frames": len(window),
    "ankle_only_peak_target_change_mm": float(np.abs(lift).max() * 1000),
    "original_fk_to_ankle_only_target_mae_mm": float(
      np.abs(raw[..., 2] - target[..., 2]).mean() * 1000
    ),
    "optimized_fk_to_same_target_mae_mm": float(
      np.abs(solved[..., 2] - target[..., 2]).mean() * 1000
    ),
    "optimized_stance_ankle_xy_mae_mm": float(
      np.linalg.norm(solved[..., :2] - raw[..., :2], axis=-1)[stance[start:stop]].mean()
      * 1000
    )
    if stance[start:stop].any()
    else None,
    "original_frames_with_native_foot_penetration_over_5mm": int(
      (native_depth[0] > 0.005).sum()
    ),
    "optimized_frames_with_native_foot_penetration_over_5mm": int(
      (native_depth[1] > 0.005).sum()
    ),
    "clearance_optimized_frames_with_native_foot_penetration_over_5mm": int(
      (native_depth[2] > 0.005).sum()
    ),
    "clearance_optimized_stance_ankle_xy_mae_mm": float(
      np.linalg.norm(clearance_ankle[..., :2] - raw[..., :2], axis=-1)[
        stance[start:stop]
      ].mean()
      * 1000
    )
    if stance[start:stop].any()
    else None,
    "nominal_stance_samples": int(support.sum()),
    "clearance_optimized_stance_hover_over_10mm": int(
      (support & (final_clearance > 0.01)).sum()
    ),
    "original_stance_hover_over_10mm": int((support & (clearance > 0.01)).sum()),
    "clearance_optimized_peak_pelvis_change_mm": float(
      np.abs(clearance_pose[:, 2] - window[:, 2]).max() * 1000
    ),
    "clearance_optimized_peak_leg_change_deg": float(
      np.rad2deg(np.abs(clearance_delta).max())
    ),
    "clearance_optimized_peak_added_leg_speed_rad_s": float(
      np.abs(np.diff(clearance_delta, axis=0)).max() * float(data["fps"].item())
    ),
    "peak_pelvis_change_mm": float(np.abs(optimized[:, 2] - window[:, 2]).max() * 1000),
    "peak_leg_change_deg": float(np.rad2deg(np.abs(delta).max())),
    "peak_added_leg_speed_rad_s": float(
      np.abs(np.diff(delta, axis=0)).max() * float(data["fps"].item())
    ),
  }, {
    "original": window,
    "optimized": optimized,
    "clearance_optimized": clearance_pose,
    "ankle_only_target": target,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("nominal", "oracle", "protocol", "out"):
    parser.add_argument(name, type=Path)
  args = parser.parse_args()
  torch.set_num_threads(1)
  nominal, oracle, protocol = [
    json.loads(path.read_text()) for path in (args.nominal, args.oracle, args.protocol)
  ]
  keys = ("starts_sha256", "terrain_seed", "rough_scale")
  if any(nominal[k] != oracle[k] for k in keys):
    raise ValueError("Expected matched evaluation inputs")
  # Use the original observed holdout field dimensions and scale.
  model, ground = collision_scene(
    nominal["terrain_seed"], rough_scale=nominal["rough_scale"]
  )
  adapter = TerrainLegReference(model, ground.height_at)
  meter = Clearance(adapter)
  clips = {Path(clip["path"]).stem: clip for clip in protocol["clips"]}
  args.out.mkdir(parents=True, exist_ok=True)
  records = {}
  for base, adapted in zip(nominal["per_motion"], oracle["per_motion"], strict=True):
    name = base["motion"]
    if adapted["motion"] != name:
      raise ValueError("Motion order does not match")
    if base["completed"] == adapted["completed"]:
      continue
    failed = adapted if base["completed"] else base
    clip = clips[name]
    source = Path(clip["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != clip["sha256"]:
      raise ValueError(f"Motion checksum mismatch for {name}")
    with np.load(source) as motion:
      result, trajectories = compare_motion(
        adapter, meter, motion, clip["local_start"] + failed["steps"] - 1, radius=10
      )
    records[name] = result
    np.savez_compressed(args.out / f"{name}.npz", **trajectories)
    print(name, json.dumps(result), flush=True)
  report = {k: nominal[k] for k in keys}
  report["records"] = records
  (args.out / "report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
  )


if __name__ == "__main__":
  main()
