"""Build a reproducible locomotion-only adaptation arm from verified clips."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256(path):
  with path.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256")
  return digest.hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("adapted", type=Path, help="Complete fixed-terrain adapted LAFAN40 directory")
  parser.add_argument("output", type=Path)
  parser.add_argument("--protocol", type=Path, default=Path("docs/experiments/lafan40_seed42.json"))
  parser.add_argument("--replacement-dir", type=Path,
                      help="Checksum-verified copies of any missing original clips")
  parser.add_argument("--other-strength", type=float, default=0.0,
                      help="Fraction of the terrain correction for non-locomotion clips, in [0, 1]")
  args = parser.parse_args()
  if not np.isfinite(args.other_strength) or not 0 <= args.other_strength <= 1:
    parser.error("--other-strength must be finite and within [0, 1]")
  if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError("Output directory must be empty")
  protocol = json.loads(args.protocol.read_text())["clips"]
  parent = json.loads((args.adapted / "manifest.json").read_text())
  if len(protocol) != 40 or len(parent["motion_clips"]) != 40:
    raise ValueError("Expected a complete LAFAN40 source and adapted set")
  args.output.mkdir(parents=True, exist_ok=True)
  rows = {}
  for entry in protocol:
    source = Path(entry["path"])
    if not source.is_file() or sha256(source) != entry["sha256"]:
      source = args.replacement_dir / source.name if args.replacement_dir else source
    if not source.is_file() or sha256(source) != entry["sha256"]:
      raise ValueError(f"Missing checksum-verified source: {entry['path']}")
    name = source.stem
    parent_row = parent["motion_clips"][name]
    adapted = args.adapted / f"{name}.npz"
    if (parent_row["source_sha256"] != entry["sha256"] or
        parent_row["frames"] != entry["frames"] or
        sha256(adapted) != parent_row["adapted_sha256"]):
      raise ValueError(f"Adapted input does not match protocol: {name}")
    use_adapted = name.startswith(("walk", "run", "sprint"))
    destination = args.output / f"{name}.npz"
    strength = 1.0 if use_adapted else args.other_strength
    if strength in (0.0, 1.0):
      shutil.copyfile(adapted if strength == 1.0 else source, destination)
    else:
      with np.load(source) as original, np.load(adapted) as terrain:
        if set(original.files) != set(terrain.files):
          raise ValueError(f"Motion fields differ: {name}")
        fields = dict(original)
        for key in ("root_pos_w", "root_lin_vel_w", "joint_pos", "joint_vel"):
          if fields[key].shape != terrain[key].shape:
            raise ValueError(f"Motion field shape differs: {name}/{key}")
          fields[key] = (fields[key] + strength * (terrain[key] - fields[key])).astype(
            fields[key].dtype
          )
      with destination.open("wb") as out:
        np.savez_compressed(out, **fields)
    rows[name] = dict(
      source_sha256=entry["sha256"],
      adapted_sha256=sha256(destination),
      frames=entry["frames"],
      fps=parent_row["fps"],
      correction_strength=strength,
    )
  manifest = dict(
    terrain_seed=parent["terrain_seed"],
    rough_scale=parent["rough_scale"],
    parent_manifest_sha256=sha256(args.adapted / "manifest.json"),
    composition="locomotion adapted; other motions use a tunable correction fraction",
    other_strength=args.other_strength,
    motion_clips=rows,
  )
  (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
  print("Composed", len(rows), "clips: 18 locomotion clips at strength 1 and 22 other clips at",
        args.other_strength)


if __name__ == "__main__":
  main()
