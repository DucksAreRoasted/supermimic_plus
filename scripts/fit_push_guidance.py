"""Fit six box-feedback gains on parallel free-base MuJoCo episodes with matched spawns."""

import argparse
import json
from pathlib import Path

import torch

from supermimic.tasks.push_rollout import PushRollout


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("checkpoint", type=Path)
  p.add_argument("motions", type=Path)
  p.add_argument("out", type=Path)
  p.add_argument("--seed", type=int, default=74)
  p.add_argument("--generations", type=int, default=4)
  p.add_argument("--candidates", type=int, default=14)
  p.add_argument("--replicates", type=int, default=12)
  a = p.parse_args()
  paths = sorted(a.motions.glob("*.npz"))
  if a.replicates % len(paths):
    p.error("Each candidate must see every reference equally often")
  n = a.candidates * a.replicates
  tasks = []
  rng = torch.Generator().manual_seed(a.seed)
  lo = torch.tensor([0.0, -0.07, 0.0, -0.08, 0.65, 0.0])
  hi = torch.tensor([1.4, 0.09, 1.5, 0.20, 1.20, 0.85])
  mean = (lo + hi) / 2
  width = (hi - lo) * 0.32
  incumbent = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
  history = []
  try:
    for m, mu in [(1.8, 0.23), (4.4, 0.39)]:
      tasks.append(
        PushRollout(a.checkpoint, paths, n, m, mu, a.seed, "cuda:0", center_x=0.96)
      )
    for i in range(a.generations):
      sampled = (mean + width * torch.randn((a.candidates, 6), generator=rng)).clamp(
        lo, hi
      )
      sampled[0] = incumbent
      sampled[1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
      scores = torch.zeros(a.candidates)
      success = torch.zeros_like(scores)
      for j, task in enumerate(tasks):
        rows, _, _ = task.run(
          a.seed + 40 * i + j,
          gap=0.12,
          offset=0.22,
          distance=0.58 if (i + j) % 2 == 0 else 0.66,
          gains=sampled.repeat_interleave(a.replicates, 0),
          feedback=True,
          scenario_repeat=a.candidates,
        )
        reward = (
          2 * rows["success"].float()
          + 0.5 * rows["bimanual"].float()
          - 0.9 * rows["other_contact"].float()
        )
        reward += (
          0.5 * rows["upright"].float()
          + 0.8 * rows["survived"].float()
          - 2 * rows["goal_error"].clamp(max=0.7)
        )
        scores += reward.reshape(a.candidates, a.replicates).mean(1) / len(tasks)
        success += rows["success"].reshape(a.candidates, a.replicates).float().mean(
          1
        ) / len(tasks)
      best = scores.argmax()
      elite = sampled[scores.topk(4).indices]
      incumbent = sampled[best]
      mean = 0.75 * elite.mean(0) + 0.25 * mean
      width = (0.7 * elite.std(0) + 0.3 * width).clamp_min((hi - lo) * 0.07)
      history.append(
        dict(
          generation=i,
          gains=incumbent.tolist(),
          score=float(scores[best]),
          success=float(success[best]),
          unconditioned_success=float(success[1]),
        )
      )
      print(json.dumps(history[-1]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
      json.dumps(
        dict(
          gains=incumbent.tolist(),
          seed=a.seed,
          train_motions=[p.name for p in paths],
          candidates=a.candidates,
          replicates=a.replicates,
          history=history,
        ),
        indent=2,
      )
      + "\n"
    )
  finally:
    for task in tasks:
      task.env.close()


if __name__ == "__main__":
  main()
