"""Generate complete terrain-adapted LAFAN motion clips for a fixed G1 height field."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from eval_g1_support_root import adapt_window
from probe_g1_terrain_reference import Clearance, collision_scene, poses_from_clip

from supermimic.tasks.terrain_reference import TerrainLegReference


def sha256(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt_clip(adapter, poses, height, fps, window=256, overlap=32):
  """Blend overlapping IK corrections so independently solved windows join smoothly."""
  if not 0 < overlap < window // 2:
    raise ValueError("Expected overlap below half the window")
  if len(poses) < 3:
    raise ValueError("A motion needs at least three frames for velocity reconstruction")
  corrected = np.zeros_like(poses)
  weights = np.zeros(len(poses))
  support_count = 0
  starts = sorted(
    set(
      (
        *range(0, len(poses) - window + 1, window - overlap),
        max(0, len(poses) - window),
      )
    )
  )
  for start in starts:
    stop = min(start + window, len(poses))
    source = poses[start:stop]
    result, meta = adapt_window(adapter, source, height, fps)
    delta = result["clearance_optimized"] - source
    ramp = np.ones(len(source))
    if start:
      ramp[:overlap] = np.linspace(0, 1, overlap + 2)[1:-1]
    if stop < len(poses):
      ramp[-overlap:] = np.minimum(
        ramp[-overlap:], np.linspace(1, 0, overlap + 2)[1:-1]
      )
    corrected[start:stop] += delta * ramp[:, None]
    weights[start:stop] += ramp
    support_count += meta["support_samples"]
  if np.any(weights == 0):
    raise RuntimeError("Reference windows left uncovered frames")
  return poses + corrected / weights[:, None], support_count


def repair_foot_collisions(adapter, poses):
  """Smoothly lift isolated residual foot contacts after window blending."""
  collision = Clearance(adapter)
  corrected = poses.copy()
  added = np.zeros(len(poses))
  for _ in range(3):
    depths = np.array([collision.contact(pose).max() for pose in corrected])
    frames = np.flatnonzero(depths > .004)
    if not len(frames):
      break
    lift = np.zeros(len(poses))
    for frame in frames:
      indices = np.arange(max(0, frame - 4), min(len(poses), frame + 5))
      bump = (depths[frame] - .0035) * np.exp(-.5 * ((indices - frame) / 1.5) ** 2)
      lift[indices] = np.maximum(lift[indices], bump)
    corrected[:, 2] += lift
    added += lift
  final_depth = max(collision.contact(pose).max() for pose in corrected)
  if final_depth > .005:
    raise RuntimeError(f"Unrepaired foot collision: {final_depth:.4f} m")
  return corrected, int(np.count_nonzero(added)), float(added.max())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument(
    "--protocol", type=Path, default=Path("docs/experiments/lafan40_seed42.json")
  )
  parser.add_argument("--terrain-seed", type=int, default=1702)
  parser.add_argument("--rough-scale", type=float, default=0.25)
  parser.add_argument("--clip", help="Generate one named motion for an initial audit")
  parser.add_argument(
    "--replacement-dir",
    type=Path,
    help="Directory with checksum-verified copies of protocol clips missing locally",
  )
  args = parser.parse_args()
  torch.set_num_threads(1)
  protocol = json.loads(args.protocol.read_text())
  clips = [
    entry
    for entry in protocol["clips"]
    if args.clip is None or Path(entry["path"]).stem == args.clip
  ]
  if not clips:
    raise ValueError("No matching motion clips")
  if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError(
      "Output directory must be empty to preserve a reproducible artifact"
    )
  args.output.mkdir(parents=True, exist_ok=True)
  model, ground = collision_scene(args.terrain_seed, rough_scale=args.rough_scale)
  adapter = TerrainLegReference(model, ground.height_at)

  def height(xy):
    return ground.height_at(torch.as_tensor(xy, dtype=torch.float32)).numpy()

  manifest = dict(
    terrain_seed=args.terrain_seed,
    rough_scale=args.rough_scale,
    generator_sha256=sha256(Path(__file__)),
    motion_clips={},
  )
  for entry in clips:
    source = Path(entry["path"])
    if sha256(source) != entry["sha256"]:
      replacement = args.replacement_dir / source.name if args.replacement_dir else None
      if (
        replacement is None
        or not replacement.is_file()
        or sha256(replacement) != entry["sha256"]
      ):
        raise ValueError(f"Motion checksum mismatch: {source}")
      source = replacement
    with np.load(source) as motion:
      fields = dict(motion)
    poses = poses_from_clip(fields, np.arange(len(fields["joint_pos"])), model)
    fps = float(fields["fps"].item())
    adapted, support_count = adapt_clip(adapter, poses, height, fps)
    adapted, repair_frames, peak_repair = repair_foot_collisions(adapter, adapted)
    joint = np.array(
      [
        model.jnt_qposadr[model.joint("robot/" + str(name)).id]
        for name in fields["joint_names"]
      ]
    )
    root_delta = adapted[:, :3] - poses[:, :3]
    joint_delta = adapted[:, joint] - poses[:, joint]
    fields["root_pos_w"] = fields["root_pos_w"] + root_delta.astype(
      fields["root_pos_w"].dtype
    )
    fields["root_lin_vel_w"] = fields["root_lin_vel_w"] + np.gradient(
      root_delta, 1 / fps, axis=0
    ).astype(fields["root_lin_vel_w"].dtype)
    fields["joint_pos"] = fields["joint_pos"] + joint_delta.astype(
      fields["joint_pos"].dtype
    )
    fields["joint_vel"] = fields["joint_vel"] + np.gradient(
      joint_delta, 1 / fps, axis=0
    ).astype(fields["joint_vel"].dtype)
    if not all(
      np.isfinite(value).all() for value in fields.values() if value.dtype.kind in "fc"
    ):
      raise ValueError(f"Nonfinite generated motion: {source}")
    destination = args.output / source.name
    with destination.open("wb") as out:
      np.savez_compressed(out, **fields)
    manifest["motion_clips"][source.stem] = dict(
      source_sha256=entry["sha256"],
      adapted_sha256=sha256(destination),
      frames=len(poses),
      fps=fps,
      support_samples_with_window_overlap=support_count,
      max_root_change_m=float(np.abs(root_delta).max()),
      max_leg_change_rad=float(np.abs(joint_delta).max()),
      max_added_joint_speed_rad_s=float(
        np.abs(np.gradient(joint_delta, 1 / fps, axis=0)).max()
      ),
      collision_repair_frames=repair_frames,
      peak_collision_repair_m=peak_repair,
    )
    print(source.stem, manifest["motion_clips"][source.stem], flush=True)
  (args.output / "manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=False)
  )


if __name__ == "__main__":
  main()
