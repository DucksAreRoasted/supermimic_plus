"""Summarize matched LAFAN40 flat evaluations and training throughput."""
import json
import sys
from pathlib import Path
from statistics import mean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(out, arms=("base", "aux", "full")):
  reports = {}
  reference = None
  for arm in arms:
    trial = json.loads((out / f"{arm}.json").read_text())
    rows = trial["per_motion"]
    panel = (trial["starts_sha256"], [(r["motion"], r["planned_frames"]) for r in rows])
    if len(rows) != 40 or (reference is not None and panel != reference):
      raise ValueError("Mismatched evaluation panel")
    reference = panel
    checkpoint = Path(trial["checkpoint"])
    try:
      num_envs = int(json.loads((checkpoint / "run_manifest.json").read_text())["num_envs"])
    except (FileNotFoundError, KeyError):
      num_envs = 4096
    run = checkpoint.parents[1]
    events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    fps = {v.step: v.value for v in events.Scalars("Perf/total_fps") if v.step > 0}
    slip = [r["stance_sole_speed_m_s"] for r in rows if r["stance_sole_speed_m_s"] is not None]
    reports[arm] = {
      "completed": trial["completed"], "failure_rate": 1 - trial["completed"] / len(rows),
      "mean_coverage": mean(r["steps"] / r["planned_frames"] for r in rows),
      "mean_joint_error": trial["means"]["error_joint_pos"],
      "mean_stance_sole_speed_m_s": mean(slip) if slip else None, "motions_with_stance": len(slip),
      "reward_per_planned_frame": trial["means"]["reward_per_reference_frame"],
      "training": [{"iteration": v.step, "interactions": (v.step + 1) * num_envs,
                    "mean_episode_reward": v.value, "fps": fps.get(v.step)}
                   for v in events.Scalars("Train/mean_reward")],
      "steady_fps_harmonic_mean": len(fps) / sum(1 / v for v in fps.values()),
    }
    for key in ("mean_abs_ankle_correction_m", "ankle_correction_step_fraction",
                "near_switch_ankle_z_error_m"):
      if key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        reports[arm][key] = mean(values) if values else None
  return reports


if __name__ == "__main__":
  print(json.dumps(summarize(Path(sys.argv[1]), sys.argv[2:] or ("base", "aux", "full")), indent=2, allow_nan=False))
