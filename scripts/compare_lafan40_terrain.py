"""Compare complete LAFAN40 motion replays against a shared physical start snapshot."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

METHODS = ("base", "render", "latent")
METRICS = ("completed", "steps", "reward_per_reference_frame", "error_body_pos",
           "mean_pelvis_xy_error_m", "termination_terms")


def compare(directory: Path):
  trials = {key: json.loads((directory / f"{key}-rough.json").read_text()) for key in METHODS}
  base = trials["base"]
  clips = [r["motion"] for r in base["per_motion"]]
  frames = [r["planned_frames"] for r in base["per_motion"]]
  if len(clips) != 40 or len(set(clips)) != 40 or base["terrain_seed"] is None:
    raise ValueError("Expected 40 unique rough-ground replays")
  for name, trial in trials.items():
    if (trial["starts_sha256"] != base["starts_sha256"] or trial["terrain_seed"] != base["terrain_seed"]
        or [r["motion"] for r in trial["per_motion"]] != clips
        or [r["planned_frames"] for r in trial["per_motion"]] != frames):
      raise ValueError(f"Starts, motion order, budget or terrain mismatch for {name}")
  rows = []
  for i, clip in enumerate(clips):
    row = {"motion": clip, "planned_frames": frames[i]}
    for name, trial in trials.items():
      motion = trial["per_motion"][i]
      row.update({f"{name}_{k}": motion[k] for k in METRICS})
      row[f"{name}_termination_terms"] = ",".join(motion["termination_terms"])
    rows.append(row)
  paired = [i for i in range(40) if all(t["per_motion"][i]["completed"] for t in trials.values())]
  summary = {}
  for name, trial in trials.items():
    items = trial["per_motion"]
    summary[name] = {
      "completed": sum(x["completed"] for x in items),
      "ref_reward": mean(x["reward_per_reference_frame"] for x in items),
      "mean_reference_coverage": mean(x["steps"] / x["planned_frames"] for x in items),
      "common_finishes": len(paired),
      "common_body_cm": 100 * mean(items[i]["error_body_pos"] for i in paired) if paired else None,
      "common_mean_pelvis_xy_m": mean(items[i]["mean_pelvis_xy_error_m"] for i in paired) if paired else None,
      "gained": [r["motion"] for r in rows if r[f"{name}_completed"] and not r["base_completed"]],
      "lost": [r["motion"] for r in rows if not r[f"{name}_completed"] and r["base_completed"]],
    }
  return rows, summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory", type=Path, help="Directory containing base-, render- and latent-rough.json")
  directory = parser.parse_args().directory
  rows, summary = compare(directory)
  with (directory / "paired-motion-results.csv").open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
  payload = json.dumps(summary, indent=2, allow_nan=False)
  (directory / "summary.json").write_text(payload + "\n")
  print(payload)


if __name__ == "__main__":
  main()
