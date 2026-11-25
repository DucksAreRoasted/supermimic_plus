"""Audit generated G1 references against their source motions and MuJoCo terrain."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from probe_g1_terrain_reference import Clearance, collision_scene, poses_from_clip

from supermimic.tasks.terrain_reference import TerrainLegReference


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("generated", type=Path)
  p.add_argument(
    "--protocol", type=Path, default=Path("docs/experiments/lafan40_seed42.json")
  )
  p.add_argument("--replacement-dir", type=Path)
  p.add_argument(
    "--stride", type=int, default=1, help="Collision-audit every Nth frame"
  )
  args = p.parse_args()
  if args.stride < 1:
    p.error("stride must be positive")
  torch.set_num_threads(1)
  manifest = json.loads((args.generated / "manifest.json").read_text())
  protocol = json.loads(args.protocol.read_text())
  model, terrain = collision_scene(
    manifest["terrain_seed"], rough_scale=manifest["rough_scale"]
  )
  adapter = TerrainLegReference(model, terrain.height_at)
  collision = Clearance(adapter)
  rows = {}
  for item in protocol["clips"]:
    name = Path(item["path"]).stem
    record = manifest["motion_clips"].get(name)
    if record is None:
      continue
    source = Path(item["path"])
    if not source.exists() or digest(source) != item["sha256"]:
      if args.replacement_dir is None:
        raise ValueError(f"Missing verified source for {name}")
      source = args.replacement_dir / source.name
    if digest(source) != item["sha256"]:
      raise ValueError(f"Source checksum mismatch for {name}")
    generated = args.generated / (name + ".npz")
    if digest(generated) != record["adapted_sha256"]:
      raise ValueError(f"Output checksum mismatch for {name}")
    with np.load(source) as raw, np.load(generated) as adapted:
      if set(raw.files) != set(adapted.files):
        raise ValueError(f"Motion schema changed for {name}")
      if len(raw["joint_pos"]) != len(adapted["joint_pos"]):
        raise ValueError(f"Frame count changed for {name}")
      for key in ("joint_names", "root_quat_w", "root_ang_vel_w", "fps"):
        np.testing.assert_array_equal(raw[key], adapted[key], err_msg=f"{name}:{key}")
      q_raw = poses_from_clip(
        raw, np.arange(0, len(raw["joint_pos"]), args.stride), model
      )
      q_new = poses_from_clip(
        adapted, np.arange(0, len(raw["joint_pos"]), args.stride), model
      )
      dq = adapted["joint_pos"] - raw["joint_pos"]
      dz = adapted["root_pos_w"][:, 2] - raw["root_pos_w"][:, 2]
      fps = float(raw["fps"].item())
      joint_velocity_error = np.abs(
        adapted["joint_vel"] - raw["joint_vel"] - np.gradient(dq, 1 / fps, axis=0)
      ).max()
      root_velocity_error = np.abs(
        adapted["root_lin_vel_w"]
        - raw["root_lin_vel_w"]
        - np.gradient(adapted["root_pos_w"] - raw["root_pos_w"], 1 / fps, axis=0)
      ).max()
      if joint_velocity_error > 2e-5 or root_velocity_error > 2e-5:
        raise ValueError(f"Velocity fields disagree with pose correction for {name}")
      depth_raw = np.array([collision.contact(pose).max() for pose in q_raw])
      depth_new = np.array([collision.contact(pose).max() for pose in q_new])
      rows[name] = dict(
        frames=len(raw["joint_pos"]),
        audited_frames=len(q_new),
        raw_penetration_over_5mm=int((depth_raw > 0.005).sum()),
        adapted_penetration_over_5mm=int((depth_new > 0.005).sum()),
        adapted_peak_penetration_m=float(depth_new.max()),
        max_added_joint_speed_rad_s=float(np.abs(np.diff(dq, axis=0)).max() * fps),
        max_added_root_speed_m_s=float(np.abs(np.diff(dz)).max() * fps),
      )
    print(name, rows[name], flush=True)
  result = dict(
    motions=len(rows),
    stride=args.stride,
    per_motion=rows,
    total_raw_penetration_over_5mm=sum(
      row["raw_penetration_over_5mm"] for row in rows.values()
    ),
    total_adapted_penetration_over_5mm=sum(
      row["adapted_penetration_over_5mm"] for row in rows.values()
    ),
  )
  (args.generated / "audit.json").write_text(
    json.dumps(result, indent=2, allow_nan=False)
  )
  print(
    "TOTAL",
    result["total_raw_penetration_over_5mm"],
    result["total_adapted_penetration_over_5mm"],
  )


if __name__ == "__main__":
  main()
