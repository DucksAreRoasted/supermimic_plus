"""Prepare terrain-feasible reward targets while preserving the original motion files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from probe_g1_terrain_reference import collision_scene, poses_from_clip
from supermimic.tasks.terrain_reference import TerrainLegReference
from train_g1_dual_head_depth import CLIPS, ROOT


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--terrain-seed", type=int, default=701)
  args = parser.parse_args()
  dest = args.output_dir
  dest.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(1)
  m, ground = collision_scene(args.terrain_seed)
  adapter = TerrainLegReference(m, ground.height_at)
  manifest = dict(terrain_seed=args.terrain_seed, terrain_height=.09, lateral_height=.04,
                  smoothing_frames=2.0, clips={}, code_sha256={
                    str(f.relative_to(ROOT)): digest(f) for f in (
                      Path(__file__).resolve(), ROOT/"scripts/probe_g1_terrain_reference.py",
                      ROOT/"src/supermimic/tasks/terrain_reference.py", ROOT/"src/supermimic/tasks/walk_terrain.py")})
  if (dest/"manifest.json").exists():
    raise FileExistsError(f"{dest} already has a published reference manifest")
  for clip in CLIPS:
    src = ROOT / "data/modelscope/LAFAN1_Retargeting_Dataset/npz" / (clip + ".npz")
    with np.load(src) as motion:
      fields = dict(motion)
    poses = poses_from_clip(fields, np.arange(len(fields["joint_pos"])), m)
    adapted = adapter.clip(poses, smoothing_frames=manifest["smoothing_frames"])
    joint = [m.jnt_qposadr[m.joint("robot/"+name).id] for name in fields["joint_names"]]
    fps = float(fields["fps"].item())
    dz = adapted[:, 2] - poses[:, 2]
    dq = adapted[:, joint] - poses[:, joint]
    fields["root_pos_w"][:, 2] += dz
    fields["root_lin_vel_w"][:, 2] += np.gradient(dz) * fps
    fields["joint_pos"] += dq
    fields["joint_vel"] += np.gradient(dq, axis=0) * fps
    if not all(np.isfinite(v).all() for v in fields.values() if v.dtype.kind in "fc"):
      raise ValueError(f"Nonfinite reward target in {clip}")
    output = dest/src.name
    tmp = dest/(clip+".partial")
    with tmp.open("wb") as file:
      np.savez_compressed(file, **fields)
    tmp.replace(output)
    manifest["clips"][clip] = dict(original_sha256=digest(src), adapted_sha256=digest(output),
                                   frames=len(poses), fps=fps,
                                   max_leg_change_rad=float(np.abs(dq).max()),
                                   max_added_joint_speed_rad_s=float(np.abs(np.diff(dq,axis=0)).max()*fps))
    print(clip, manifest["clips"][clip], flush=True)
  tmp = dest/"manifest.tmp"
  tmp.write_text(json.dumps(manifest, indent=2,allow_nan=False)+"\n")
  tmp.replace(dest/"manifest.json")


if __name__ == "__main__":
  main()
