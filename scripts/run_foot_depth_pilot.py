"""Reproduce the two-arm GPU smoke, adapter pilot and paired rough-terrain evaluation."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from validate_depth_mimic import CASES

ROOT = Path(__file__).resolve().parents[1]
BASE = (
  ROOT
  / "logs/SuperMimic-GenCoord-G1/terrain-5cm-actor-world-cross0.025-adapted-384it-seed42/2026-11-20_16-41-43"
)
CLIPS = ROOT / "data/modelscope/LAFAN1_Retargeting_Dataset/npz"
TRAIN = [
  ("walk1_subject5", 271, 0.09, 0.04, 701),
  ("walk2_subject4", 389, 0.11, 0.05, 853),
]
MODES = ("concat", "query")


def run(script, args, log):
  log.parent.mkdir(parents=True, exist_ok=True)
  env = dict(
    os.environ,
    PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    SUPERMIMIC_DISABLE_COMPILE="1",
    TORCH_COMPILE_DISABLE="1",
    MUJOCO_GL="egl",
    OMP_NUM_THREADS="1",
    PYTHONUNBUFFERED="1",
  )
  print(script, log.parent.name, log.name, flush=True)
  with log.open("w") as stream:
    subprocess.run(
      [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
      cwd=ROOT,
      env=env,
      stdout=stream,
      stderr=subprocess.STDOUT,
      check=True,
    )


def evaluate(
  base, case, directory, count, steps, *, stage, weights=None, frame_range=(4000, 11800)
):
  clip, seed, height, lateral, layout = case
  name = weights.stem if weights else "depth"
  out = directory / f"{name}.json"
  options = [
    base,
    CLIPS / f"{clip}.npz",
    out,
    "--perception",
    "depth",
    "--num-envs",
    count,
    "--steps",
    steps,
    "--seed",
    seed,
    "--height",
    height,
    "--lateral-height",
    lateral,
    "--landscape-seed",
    layout,
    "--depth-noise-m",
    0.005,
    "--depth-dropout",
    0.15,
    "--solver-iterations",
    60,
    "--starts-file",
    directory / "starts.npz",
  ]
  if stage == "holdout":
    options += [
      "--window-search-frames",
      frame_range[1],
      "--first-start-frame",
      frame_range[0],
      "--disjoint-windows",
    ]
  else:
    options += ["--window-search-frames", 3600]
  if weights:
    options += ["--fusion-checkpoint", weights]
  if stage == "banks":
    options += ["--record-fusion-bank", directory / "bank.pt"]
  run("eval_walk_terrain.py", options, directory / f"{name}-eval.log")


def verify_pair(directory, weights):
  arms = [json.loads((directory / f"{m}.json").read_text()) for m in MODES]
  for mode, arm in zip(MODES, arms, strict=True):
    if (
      arm["fusion_mode"] != mode
      or arm["external_joint_compensation"]
      or arm["perception"] != "depth"
      or arm["fusion_sha256"]
      != hashlib.sha256((weights / f"{mode}.pt").read_bytes()).hexdigest()
      or not arm["mean_leg_action_delta_masking_fusion_depth"] > 0
    ):
      raise ValueError(
        f"Unexpected policy or disconnected depth branch in {directory}/{mode}"
      )
  for k in (
    "clip_sha256",
    "seed",
    "landscape_seed",
    "steps",
    "height",
    "lateral_height",
    "depth_noise_m",
    "depth_dropout",
  ):
    if arms[0][k] != arms[1][k]:
      raise ValueError(f"Unpaired {k} at {directory}")
  # Derived body velocities can differ by one float32 rounding step on GPU.
  # qpos, qvel, motion frame and physical context still have to match exactly.
  with np.load(directory / "starts.npz") as start:
    common = np.concatenate([start["full_state"], start["context"]], axis=-1)
    start_hash = hashlib.sha256(common.tobytes()).hexdigest()
    for arm in arms:
      diff = arm["initial_max_abs_difference"]
      if not diff and arm["start_state_sha256"] == start_hash:
        continue  # The run that wrote the shared snapshot.
      if (
        any(diff.get(k) != 0 for k in ("frames", "qpos", "qvel", "context"))
        or not 0 <= diff.get("full_state", float("inf")) <= 1e-6
      ):
        raise ValueError(f"Physical starts differ in {directory}")
  if [r["start"] for r in arms[0]["windows_detail"]] != [
    r["start"] for r in arms[1]["windows_detail"]
  ]:
    raise ValueError(f"Unpaired starts in {directory}")
  return arms


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "stage", choices=("all", "banks", "smoke", "pilot", "holdout", "summarize")
  )
  parser.add_argument("--base", type=Path, default=BASE)
  parser.add_argument("--out", type=Path, default=ROOT / "outputs/foot-depth-pilot")
  args = parser.parse_args()
  out, base = args.out.resolve(), args.base.resolve()
  stages = (
    ("banks", "smoke", "pilot", "holdout") if args.stage == "all" else (args.stage,)
  )
  for stage in stages:
    if stage == "banks":
      for case in TRAIN:
        evaluate(base, case, out / "banks" / case[0], 24, 180, stage=stage)
    elif stage in ("smoke", "pilot"):
      banks = [out / "banks" / case[0] / "bank.pt" for case in TRAIN]
      if stage == "pilot":
        smoke = out / "smoke"
        verify_pair(smoke, smoke)
        manifest = json.loads((smoke / "training.json").read_text())
        for filename, field in (
          ("actor.pt", "base_sha256"),
          ("world_model.pt", "world_model_sha256"),
          ("critic.pt", "critic_sha256"),
        ):
          if (
            hashlib.sha256((base / filename).read_bytes()).hexdigest()
            != manifest[field]
          ):
            raise ValueError(f"{filename} differs from the GPU smoke")
        if sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in banks) != sorted(
          manifest["bank_sha256"].values()
        ):
          raise ValueError("Training banks differ from the GPU smoke")
      directory = out / stage
      options = [base, directory, *banks]
      if stage == "smoke":
        options += ["--updates", 5, "--batch", 12, "--horizon", 6]
      run("train_foot_depth_actor.py", options, directory / "train.log")
      if stage == "smoke":
        for mode in MODES:
          evaluate(
            base,
            TRAIN[0],
            directory,
            8,
            48,
            stage=stage,
            weights=directory / f"{mode}.pt",
          )
        verify_pair(directory, directory)
    else:
      aggregate = {m: [] for m in MODES}
      # Exclude walk1_subject1, which was used to adapt the inherited terrain checkpoint.
      for suite, count in (("confirmation", 12), ("prospective", 16)):
        for case in CASES[suite]:
          clip = case[0]
          if clip == "walk1_subject1":
            continue
          directory = out / "holdout" / clip
          if stage == "holdout":
            directory.mkdir(parents=True, exist_ok=True)
            old_starts = ROOT / "outputs/perceptive_walk" / suite / clip / "starts.npz"
            if old_starts.exists() and not (directory / "starts.npz").exists():
              shutil.copyfile(old_starts, directory / "starts.npz")
            for mode in MODES:
              evaluate(
                base,
                case,
                directory,
                count,
                300,
                stage=stage,
                weights=out / "pilot" / f"{mode}.pt",
              )
          pair = verify_pair(directory, out / "pilot")
          for mode, arm in zip(MODES, pair, strict=True):
            aggregate[mode].extend(arm["windows_detail"])
      result = {}
      pairs = list(zip(*aggregate.values(), strict=True))
      shared_steps = [min(a["steps"], b["steps"]) for a, b in pairs]
      for mode, rows in aggregate.items():
        result[mode] = dict(
          windows=len(rows),
          successes=sum(r["success"] for r in rows),
          survived=sum(r["survived"] for r in rows),
          body_error_cm=mean(r["mean_body_error_cm"] for r in rows),
          common_prefix_body_error_cm=mean(
            mean(r["body_error_trace_cm"][:t])
            for r, t in zip(rows, shared_steps, strict=True)
          ),
          upper_body_error_cm=mean(r["upper_body_error_cm"] for r in rows),
          last_root_xy_error_m=mean(r["last_root_xy_error_m"] for r in rows),
          reward_per_planned_step=mean(r["reward_per_planned_step"] for r in rows),
        )
      result["paired"] = {
        key: sum((a["success"], b["success"]) == flags for a, b in pairs)
        for key, flags in {
          "both": (True, True),
          "concat_only": (True, False),
          "query_only": (False, True),
          "neither": (False, False),
        }.items()
      }
      (out / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
      )
      print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
