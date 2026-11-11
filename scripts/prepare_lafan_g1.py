"""Download/convert ModelScope LAFAN1 G1 clips into mjlab NPZ motion files."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from mjlab.scripts.csv_to_npz import MotionLoader


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--root", type=Path, default=Path("data/modelscope/LAFAN1_Retargeting_Dataset")
  )
  parser.add_argument(
    "--clips", type=int, default=0, help="Convert only the first N clips; 0 means all"
  )
  parser.add_argument("--input-fps", type=float, default=30.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  args = parser.parse_args()

  args.root.parent.mkdir(parents=True, exist_ok=True)
  if not args.root.exists() or not any(args.root.glob("**/*.csv")):
    subprocess.run(
      [
        "modelscope",
        "download",
        "--dataset",
        "unitreerobotics/LAFAN1_Retargeting_Dataset",
        "--local_dir",
        str(args.root),
      ],
      check=True,
    )

  csv_files = sorted((args.root / "g1").glob("*.csv"))
  if args.clips > 0:
    csv_files = csv_files[: args.clips]
  if not csv_files:
    raise FileNotFoundError(f"No G1 CSV clips found under {args.root / 'g1'}")

  output_dir = args.root / "npz"
  output_dir.mkdir(exist_ok=True)
  for csv_file in csv_files:
    output_file = output_dir / f"{csv_file.stem}.npz"
    if output_file.exists():
      print(f"[skip] {output_file}")
      continue
    motion = MotionLoader(
      str(csv_file),
      input_fps=args.input_fps,
      output_fps=args.output_fps,
      device="cpu",
    )
    np.savez(output_file, **motion.arrays())
    print(f"[ok] {csv_file.name} -> {output_file}")

  print(f"Prepared {len(csv_files)} G1 clip(s) in {output_dir}")


if __name__ == "__main__":
  main()
