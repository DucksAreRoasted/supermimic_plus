"""Evaluate a free-base G1 pushing a movable box and optionally save simulator transitions."""

import argparse
import json
from pathlib import Path

import torch

from supermimic.models_push import PushPolicy
from supermimic.tasks.push_rollout import PushRollout


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("checkpoint", type=Path)
  p.add_argument("motions", type=Path)
  p.add_argument("output", type=Path)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--num-envs", type=int, default=32)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--mass", type=float, default=2.0)
  p.add_argument("--friction", type=float, default=0.30)
  p.add_argument(
    "--offset", type=float, default=0.12, help="Full lateral spawn range in meters"
  )
  p.add_argument(
    "--gap", type=float, default=0.10, help="Full longitudinal spawn range in meters"
  )
  p.add_argument("--center-x", type=float, default=0.76)
  p.add_argument("--lateral-bias", type=float, default=0.0)
  p.add_argument("--feedback", action="store_true")
  p.add_argument(
    "--stop-at-goal",
    action="store_true",
    help="Adapt all ten reference frames to measured box progress",
  )
  p.add_argument(
    "--push-force",
    type=float,
    default=0.0,
    help="Maximum horizontal palm assistance in newtons",
  )
  p.add_argument(
    "--force-adaptive",
    action="store_true",
    help="Blend corrections using measured palm resistance",
  )
  p.add_argument(
    "--gains",
    type=float,
    nargs=6,
    help="Hand, pressure, foot, lean, phase rate, braking",
  )
  p.add_argument("--terrain-height", type=float, default=0.0)
  p.add_argument("--terrain-phase", type=int, default=0)
  p.add_argument("--grade", type=float, default=0.0, help="Planar rise divided by run")
  p.add_argument("--distance", type=float, default=0.67)
  p.add_argument("--steps", type=int, default=145)
  p.add_argument("--explore", type=float, default=0.0)
  p.add_argument("--record", action="store_true")
  p.add_argument("--video", type=Path, help="Render the first measured episode to MP4")
  p.add_argument(
    "--policy",
    type=Path,
    help="Trained object-aware residual on top of the tracking actor",
  )
  args = p.parse_args()
  files = sorted(args.motions.glob("*.npz"))
  if not files:
    p.error("No G1 lane references")
  task = PushRollout(
    args.checkpoint,
    files,
    args.num_envs,
    args.mass,
    args.friction,
    args.seed,
    args.device,
    args.center_x,
    args.lateral_bias,
    args.terrain_height,
    args.terrain_phase,
    args.grade,
  )
  try:
    policy = None
    if args.policy:
      policy = PushPolicy().to(args.device)
      policy.load_state_dict(
        torch.load(args.policy, map_location=args.device, weights_only=True)
      )
      policy.eval()
    metrics, start, data = task.run(
      args.seed,
      args.steps,
      args.gap,
      args.offset,
      args.distance,
      policy=policy,
      explore=args.explore,
      record=args.record,
      feedback=args.feedback or args.gains is not None,
      gains=args.gains,
      force_adaptive=args.force_adaptive,
      push_force=args.push_force,
      stop_at_goal=args.stop_at_goal,
      video_path=args.video,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if data:
      torch.save({k: v.cpu() for k, v in data.items()}, args.output / "transitions.pt")
    torch.save(start, args.output / "initial.pt")
    result = dict(
      motions=[v.name for v in files],
      mass=args.mass,
      friction=args.friction,
      seed=args.seed,
      distance=args.distance,
      steps=args.steps,
      envs=args.num_envs,
      center_x=args.center_x,
      lateral_bias=args.lateral_bias,
      gap=args.gap,
      offset=args.offset,
      feedback=args.feedback or args.gains is not None,
      gains=args.gains,
      force_adaptive=args.force_adaptive,
      push_force=args.push_force,
      stop_at_goal=args.stop_at_goal,
      checkpoint=str(args.checkpoint),
      policy=str(args.policy) if args.policy else None,
      terrain_height=args.terrain_height,
      terrain_phase=args.terrain_phase,
      grade=args.grade,
      means={k: float(v.float().mean()) for k, v in metrics.items() if k != "lane"},
      per_episode={k: v.tolist() for k, v in metrics.items()},
    )
    (args.output / "eval.json").write_text(
      json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result["means"], indent=2))
  finally:
    task.env.close()


if __name__ == "__main__":
  main()
