"""Run the frozen depth-mimic policy and controls on paired raw LAFAN walking windows."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# clip, start seed, longitudinal/irregular amplitude, lateral amplitude, phase or landscape seed
CASES = {
  "development": [("walk1_subject2", 161, 0.08, 0.035, 14)],
  "periodic": [
    ("walk2_subject3", 139, 0.085, 0.03, 37),
    ("walk3_subject1", 223, 0.09, 0.04, 61),
    ("walk3_subject4", 347, 0.10, 0.035, 49),
  ],
  "irregular": [
    ("walk1_subject5", 321, 0.105, 0.045, 817),
    ("walk3_subject2", 419, 0.13, 0.05, 957),
  ],
  "long": [
    ("walk3_subject3", 551, 0.11, 0.05, 1101),
    ("walk2_subject1", 691, 0.12, 0.045, 1217),
  ],
  "long-extra": [
    ("walk4_subject1", 733, 0.12, 0.05, 1319),
    ("walk3_subject5", 827, 0.11, 0.055, 1409),
  ],
  "prospective": [
    ("walk1_subject1", 997, 0.11, 0.045, 1777),
    ("walk2_subject3", 1123, 0.105, 0.05, 1861),
    ("walk3_subject4", 1237, 0.125, 0.045, 1999),
  ],
  "confirmation": [
    ("walk2_subject1", 1429, 0.12, 0.04, 2131),
    ("walk3_subject2", 1531, 0.105, 0.055, 2273),
    ("walk3_subject5", 1657, 0.115, 0.045, 2393),
  ],
  "flat": [("walk2_subject4", 147, 0.075, 0.03, 67)],
}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("suite", choices=CASES)
  parser.add_argument(
    "--modes",
    nargs="+",
    default=["depth", "depth-contact-stride", "depth-contact-stride-shuffled"],
  )
  parser.add_argument(
    "--output", type=Path, default=ROOT / "outputs" / "depth-mimic-validation"
  )
  parser.add_argument(
    "--starts-root",
    type=Path,
    help="Verify against a previous suite's saved physical starts",
  )
  args = parser.parse_args()
  rough = args.suite in (
    "irregular",
    "long",
    "long-extra",
    "prospective",
    "confirmation",
  )
  counts = {
    "development": (24, 180),
    "flat": (24, 200),
    "periodic": (32, 200),
    "irregular": (32, 200),
    "long": (32, 400),
    "long-extra": (32, 400),
    "prospective": (16, 300),
    "confirmation": (12, 300),
  }
  num_envs, steps = counts[args.suite]
  for clip, seed, height, lateral, layout in CASES[args.suite]:
    directory = (args.output / args.suite / clip).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.starts_root:
      previous = args.starts_root / args.suite / clip / "starts.npz"
      snapshot = directory / "starts.npz"
      if not snapshot.exists():
        shutil.copyfile(previous, snapshot)
      elif snapshot.read_bytes() != previous.read_bytes():
        raise ValueError(f"Initial-state file differs from {previous}")
    for mode in args.modes:
      archive = mode == "archive"
      out = directory / "archived" if archive else directory
      out.mkdir(exist_ok=True)
      options = [
        "--num-envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--height",
        str(height),
        "--lateral-height",
        str(lateral),
        "--landscape-seed" if rough else "--phase",
        str(layout),
        "--starts-file",
        str(directory / "starts.npz"),
        "--foot-gain",
        "1.5",
        "--steering-gain",
        ".45",
        "--balance-gain",
        ".5",
      ]
      if args.suite in ("prospective", "confirmation"):
        options += [
          "--first-start-frame",
          "4000",
          "--window-search-frames",
          "11800",
          "--disjoint-windows",
        ]
      if rough:
        options += ["--depth-noise-m", ".005", "--depth-dropout", ".15"]
      if args.suite == "flat":
        options += ["--flat"]
      if archive:
        default = (
          ROOT
          / "logs/SuperMimic-GenCoord-G1/g1-shared-focus04-nograph-reset-safe-simweight0-warm24k-plus16k-seed42/2026-11-20_12-29-41"
        )
        options += ["--driver", os.environ.get("G1_BASELINE_CHECKPOINT", str(default))]
      env = dict(
        os.environ,
        G1_WALK_MOTION=f"data/modelscope/LAFAN1_Retargeting_Dataset/npz/{clip}.npz",
        G1_SCENE_OUTPUT=str(out),
        G1_PERCEPTION="depth-blind" if archive else mode,
      )
      print(f"{args.suite} {clip} {mode} -> {out}", flush=True)
      with (out / f"{mode}.log").open("w") as log:
        subprocess.run(
          ["bash", "scripts/run_g1_scene.sh", "perceptive", *options],
          cwd=ROOT,
          env=env,
          stdout=log,
          stderr=subprocess.STDOUT,
          check=True,
        )


if __name__ == "__main__":
  main()
