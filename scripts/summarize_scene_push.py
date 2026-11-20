"""Check the paired G1 push benchmark and summarize counts without pooling training runs."""

import argparse
import json
from pathlib import Path

import torch


def summarize(root):
  groups = {}
  variants = ("control", "paced", "candidate")
  for scene in ("flat", "grade", "ledges", "transfer"):
    rows = []
    for seed in (1221, 1223) if scene == "transfer" else (1201, 1202, 1203):
      prefix = f"transfer-{seed}" if scene == "transfer" else f"release-{seed}-{scene}"
      starts = []
      reports = []
      for variant in variants:
        path = root / (
          f"ablation-{seed}-{scene}-pacing"
          if variant == "paced"
          else f"{prefix}-{variant}"
        )
        starts.append(torch.load(path / "initial.pt", weights_only=True))
        report = json.loads((path / "eval.json").read_text())
        reports.append(report)
      for initial in starts[1:]:
        torch.testing.assert_close(starts[0], initial, atol=1.0e-6, rtol=0.0)
      for key in (
        "motions",
        "mass",
        "friction",
        "seed",
        "distance",
        "steps",
        "envs",
        "center_x",
        "lateral_bias",
        "gap",
        "offset",
        "terrain_height",
        "terrain_phase",
        "grade",
      ):
        if any(d[key] != reports[0][key] for d in reports[1:]):
          raise ValueError(f"Mismatched evaluation setting {prefix}: {key}")
      result = {}
      for key, d in zip(variants, reports, strict=True):
        ep = d["per_episode"]
        result[key] = {
          k: sum(ep[k])
          for k in ("success", "parked", "survived", "bimanual", "other_contact")
        }
        result[key]["both_palms_simultaneous"] = sum(
          x > 0 for x in ep["simultaneous_steps"]
        )
        result[key]["goal_error_m"] = d["means"]["goal_error"]
        if any(p >= 0.001 for p in ep["start_penetration_m"]):
          raise ValueError(f"Invalid initial overlap in {prefix}")
      rows.append(
        dict(
          seed=seed,
          mass=d["mass"],
          mu=d["friction"],
          distance=d["distance"],
          episodes=d["envs"],
          max_initial_difference=max(
            (initial - starts[0]).abs().max().item() for initial in starts[1:]
          ),
          **result,
        )
      )
    groups[scene] = dict(conditions=rows, episodes=sum(r["episodes"] for r in rows))
    for key in variants:
      groups[scene][key] = {
        field: sum(r[key][field] for r in rows)
        for field in (
          "success",
          "parked",
          "survived",
          "bimanual",
          "other_contact",
          "both_palms_simultaneous",
        )
      }
  return groups


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("root", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  out = summarize(args.root)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
  for scene, data in out.items():
    print(
      scene,
      data["episodes"],
      data["control"]["success"],
      data["paced"]["success"],
      data["candidate"]["success"],
      data["candidate"]["parked"],
    )
