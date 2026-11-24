"""Compare paired on/off leg modulation replays for one checkpoint."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def summarize(rows):
  stance = sum(r["stance_foot_seconds"] for r in rows)
  landings = sum(r["landings"] for r in rows)
  failures = sum(r["landings_followed_by_failure_5steps"] for r in rows)
  return {
    "motions": len(rows), "completed": sum(r["completed"] for r in rows),
    "mean_coverage": mean(r["steps"] / r["planned_frames"] for r in rows),
    "stance_speed_m_s": sum((r["stance_sole_speed_m_s"] or 0) * r["stance_foot_seconds"]
                            for r in rows) / stance if stance else None,
    "landings": landings, "landings_followed_by_failure_5steps": failures,
    "landing_failure_rate": failures / landings if landings else None,
    "mean_abs_leg_delta": mean(r["mean_abs_applied_leg_delta"] for r in rows),
    "mean_joint_error": mean(r["error_joint_pos"] for r in rows),
  }


def compare(directory):
  summary, details = {}, []
  for terrain in ("flat", "easy"):
    on, off = (json.loads((directory / f"{terrain}-{arm}.json").read_text()) for arm in ("on", "off"))
    if (on["leg_ablation"], off["leg_ablation"]) != ("on", "off") or any(
      on[key] != off[key] for key in ("starts_sha256", "actor_sha256", "rough_scale", "terrain_seed")
    ):
      raise ValueError("Mismatched checkpoint, terrain or start states")
    rows_on, rows_off = on["per_motion"], off["per_motion"]
    if len(rows_on) != 40 or len({r["motion"] for r in rows_on}) != 40:
      raise ValueError("Expected LAFAN40")
    for a, b in zip(rows_on, rows_off, strict=True):
      if (a["motion"], a["planned_frames"]) != (b["motion"], b["planned_frames"]):
        raise ValueError("Mismatched motion pair")
      for mode, row in (("on", a), ("off", b)):
        details.append({"terrain": terrain, "modulation": mode, **row})
    summary[terrain] = {"on": summarize(rows_on), "off": summarize(rows_off),
                        "gained": [a["motion"] for a,b in zip(rows_on, rows_off) if a["completed"] and not b["completed"]],
                        "lost": [a["motion"] for a,b in zip(rows_on, rows_off) if b["completed"] and not a["completed"]]}
  with (directory / "per-motion.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=details[0])
    writer.writeheader()
    writer.writerows(details)
  (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory", type=Path)
  compare(parser.parse_args().directory)
