"""Check pairing and summarize the G1 depth-mimic experiments, including ablations."""

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import mean

import numpy as np
from validate_depth_mimic import CASES, ROOT

LEGACY = {"periodic": "heldout", "long-extra": "stride-holdout", "flat": "flat_depth"}
MEANS = {
  "reward_per_planned_step": "mean_reward_per_planned_step",
  "body_cm": "mean_body_error_cm",
  "upper_body_cm": "mean_upper_body_error_cm",
}


def digest(path):
  with path.open("rb") as file:
    return hashlib.file_digest(file, "sha256").hexdigest()


def metrics(report):
  return {
    **{k: report[k] for k in ("windows", "successes", "survived")},
    **{k: report[key] for k, key in MEANS.items()},
    "last_root_xy_m": mean(
      row["last_root_xy_error_m"] for row in report["windows_detail"]
    ),
  }


def summarize(source, only=None):
  cases, totals, weights = {}, {}, {}
  for suite, scenarios in CASES.items():
    if suite == "development" or (only is not None and suite != only):
      continue
    directory = source / suite
    if not directory.exists():
      directory = source / LEGACY.get(suite, suite)
    groups = defaultdict(list)
    for clip, *_ in scenarios:
      folder = directory / clip
      if not folder.exists() and suite == "flat":
        folder = source / LEGACY["flat"]
      if not (folder / "starts.npz").is_file():
        raise ValueError(f"Missing paired starts: {folder}")
      with np.load(folder / "starts.npz") as snapshot:
        starts = snapshot["frames"].tolist()
      files = sorted(folder.glob("*.json")) + sorted(folder.glob("archived/*.json"))
      reports = {}
      for path in files:
        r = json.loads(path.read_text())
        rows = r["windows_detail"]
        if starts != [row["start"] for row in rows] or r["windows"] != len(rows):
          raise ValueError(f"Different windows in {path}")
        if r["successes"] != sum(row["success"] for row in rows) or r[
          "survived"
        ] != sum(row["survived"] for row in rows):
          raise ValueError(f"Inconsistent outcome counts in {path}")
        for key in ("qpos", "qvel", "frames"):
          if r["initial_max_abs_difference"].get(key, 0) != 0:
            raise ValueError(f"Different initial {key} in {path}")
        for key in ("context", "full_state"):
          if r["initial_max_abs_difference"].get(key, 0) > 1e-6:
            raise ValueError(f"Different initial {key} in {path}")
        if r.get("disjoint_windows") and any(
          b - a <= r["steps"] for a, b in pairwise(starts)
        ):
          raise ValueError(f"Overlapping disjoint windows in {path}")
        mode = "archive" if path.parent.name == "archived" else path.stem
        reports[mode] = r
        for key in ("checkpoint", "driver"):
          checkpoint = Path(r[key])
          if str(checkpoint) not in weights:
            weights[str(checkpoint)] = {
              name: digest(ROOT / checkpoint / name)
              for name in ("actor.pt", "world_model.pt")
            }
      if "depth" not in reports:
        raise ValueError(f"Missing camera-only control in {folder}")
      base = reports["depth"]
      keys = (
        "height",
        "lateral_height",
        "landscape_seed",
        "phase_cells",
        "seed",
        "steps",
        "window_search_frames",
        "first_start_frame",
        "disjoint_windows",
        "depth_noise_m",
        "depth_dropout",
        "solver_iterations",
        "solver_ls_iterations",
        "flat",
        "clip_sha256",
      )
      settings = {k: base.get(k, None if k == "landscape_seed" else 0) for k in keys}
      case = {
        "settings": settings,
        "start_frames": starts,
        "start_snapshot_sha256": digest(folder / "starts.npz"),
        "modes": {},
      }
      for mode, r in reports.items():
        if mode != "archive" and r["driver"] != base["driver"]:
          raise ValueError(f"Actor mismatch: {suite}/{clip}/{mode}")
        if {
          k: r.get(k, None if k == "landscape_seed" else 0) for k in keys
        } != settings:
          raise ValueError(f"Protocol mismatch: {suite}/{clip}/{mode}")
        values = metrics(r)
        success = [row["start"] for row in r["windows_detail"] if row["success"]]
        base_success = {
          row["start"] for row in base["windows_detail"] if row["success"]
        }
        values["wins_vs_depth"] = len(set(success) - base_success)
        values["losses_vs_depth"] = len(base_success - set(success))
        groups[mode].append(values)
        case["modes"][mode] = {
          **values,
          "driver": r["driver"],
          "controller_gains": {
            key: r.get(key)
            for key in ("foot_gain", "steering_gain", "balance_gain", "stride_gain")
          },
          "success_frames": success,
          "camera_valid_fraction": r["scan"]["valid_fraction"],
          "camera_height_mae_cm": r["scan"]["height_mae_cm"],
          "initial_max_abs_difference": r["initial_max_abs_difference"],
        }
      cases[f"{suite}/{clip}"] = case
    totals[suite] = {}
    for mode, values in groups.items():
      if len(values) != len(scenarios):  # Only report complete cohorts.
        continue
      summed = {
        k: sum(v[k] for v in values)
        for k in (
          "windows",
          "successes",
          "survived",
          "wins_vs_depth",
          "losses_vs_depth",
        )
      }
      avg = {
        k: sum(v[k] * v["windows"] for v in values) / summed["windows"]
        for k in (*MEANS, "last_root_xy_m")
      }
      totals[suite][mode] = {**summed, **avg}
  development = {
    group: {
      p.parent.name: metrics(json.loads(p.read_text()))
      for p in sorted((source / group).glob("*/*.json"))
    }
    for group in ("dev", "stride-dev")
  }
  return {
    "weights": weights,
    "development": development,
    "totals": totals,
    "cases": cases,
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  data = summarize(args.source)
  repeats = {}
  for folder in sorted(p for p in args.source.glob("repeat-[0-9]*") if p.is_dir()):
    replica = summarize(folder, only="confirmation")
    for name, case in replica["cases"].items():
      original = data["cases"][name]
      for field in ("start_snapshot_sha256", "settings", "start_frames"):
        if case[field] != original[field]:
          raise ValueError(f"Unpaired repeat: {folder}/{name}/{field}")
    required = {"depth", "depth-contact-stride", "depth-contact-stride-shuffled"}
    if not required.issubset(replica["totals"]["confirmation"]):
      raise ValueError(f"Incomplete confirmation repeat: {folder}")
    repeats[folder.name] = {
      "totals": replica["totals"]["confirmation"],
      "cases": replica["cases"],
    }
  data["confirmation_repeats"] = repeats
  args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
