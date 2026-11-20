"""Optimize a bounded G1 action residual through learned hand-box dynamics."""

import argparse
import json
from pathlib import Path

import torch
from fit_push_dynamics import load_runs, sample_paths
from mjlab.utils.lab_api.math import quat_apply_inverse
from torch.nn import functional as F

from supermimic.models import build_mlp
from supermimic.models_push import ContactDynamics, PushPolicy
from supermimic.representations import extract_actor_goal
from supermimic.tasks.push_rollout import PushRollout
from supermimic.tasks.task_spec import build_actor_obs_td


def box_distance(points, box, size):
  q = box[:, None, 3:7].expand(*points.shape[:-1], 4)
  local = quat_apply_inverse(q, points - box[:, None, :3])
  delta = local.abs() - size
  return delta.clamp_min(0.0).norm(dim=-1) + delta.max(-1).values.clamp_max(0.0)


def update(task, dynamics, policy, critic, data, optim, horizon, batch):
  form = task.wm.representation
  t, b = sample_paths(data, horizon, batch)
  aux, rep = (
    form.extract_aux_state(data["full"][t, b]),
    form.represent(data["full"][t, b]),
  )
  box, palms, physics = data["box"][t, b], data["palms"][t, b], data["physics"][t, b]
  frame, ctx, goal = data["frame"][t, b], data["ctx"][t, b], data["goal"][t, b]
  clip_end = (
    task.motion.motion_ends[
      torch.searchsorted(task.motion.motion_ends, frame, right=True)
    ]
    - 1
  )
  # Random goals make the continuation depend on the requested destination.
  goal = goal + (torch.rand_like(goal) - 0.5) * goal.new_tensor([0.18, 0.10])
  initial_err = (goal - box[:, :2]).norm(dim=-1)
  last_action = data["action"][(t - 1).clamp_min(0), b]
  toes_idx = [
    task.command.cfg.body_names.index(s + "_ankle_roll_link") for s in ("left", "right")
  ]
  cost = torch.zeros_like(initial_err)
  for k in range(horizon):
    target_frame = torch.minimum(frame + k, clip_end)
    wm_goal, _ = form.compute_wm_goal(task.motion, target_frame, aux)
    base = task.actor(
      build_actor_obs_td({"motion": extract_actor_goal(wm_goal)}, rep.unsqueeze(1))
    )
    residual = policy(rep, aux, box, palms, goal)
    action = base + residual
    aux, rep, box, palms = dynamics.step(
      task, aux, rep, action, ctx, box, palms, physics
    )
    joints = form.joint_state(rep)
    ref = task.motion.body_state(target_frame)
    tracking = form.compute_wm_reward_from_reference(
      rep, aux, joints, action, last_action, *ref
    )
    pos, rot, _, _ = form.world_frame_bodies(aux, rep)
    toe = pos[:, toes_idx] + (rot[:, toes_idx] @ pos.new_tensor([0.13, 0.0, -0.025]))
    guard = F.softplus((0.06 - box_distance(toe, box, dynamics.size)) / 0.025).mean(-1)
    local = quat_apply_inverse(
      box[:, None, 3:7].expand(-1, 2, -1), palms - box[:, None, :3]
    )
    targets = box.new_tensor([[-0.38, 0.14, 0.35], [-0.38, -0.14, 0.35]])
    hand_error = (local - targets).square().sum(-1).mean(-1)
    distance_sq = (goal - box[:, :2]).square().sum(-1)
    tilt_cost = box[:, 4:6].square().sum(-1)
    # Tracking returns already include dt. Each other running cost is expressed per second.
    cost += 0.99**k * (
      -0.65 * tracking
      + dynamics.dt
      * (
        0.22 * guard
        + 0.7 * hand_error
        + 2.5 * distance_sq
        + (0.05 * residual.square().mean(-1))
        + 1.5 * tilt_cost
      )
    )
    last_action = action
  wm_goal, _ = form.compute_wm_goal(
    task.motion, torch.minimum(frame + horizon, clip_end), aux
  )
  value = critic(torch.cat([rep, wm_goal, ctx], -1)).squeeze(-1)
  terminal_error = (goal - box[:, :2]).norm(dim=-1)
  loss = (
    cost + 1.5 * (terminal_error - initial_err) - 0.08 * (0.99**horizon) * value
  ).mean()
  if not torch.isfinite(loss):
    raise ValueError("Nonfinite imagined push objective")
  optim.zero_grad()
  loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(
    policy.parameters(), 1.0, error_if_nonfinite=True
  )
  optim.step()
  return dict(
    loss=loss.item(),
    grad_norm=norm.item(),
    predicted_progress_m=(initial_err - terminal_error).mean().item(),
  )


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("checkpoint", type=Path)
  p.add_argument("motions", type=Path)
  p.add_argument("dynamics", type=Path)
  p.add_argument("out", type=Path)
  p.add_argument("--train", nargs="+", type=Path, required=True)
  p.add_argument("--updates", type=int, default=120)
  p.add_argument("--horizon", type=int, default=8)
  p.add_argument("--batch", type=int, default=64)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--lr", type=float, default=0.0004)
  args = p.parse_args()
  motions = sorted(args.motions.glob("*.npz"))
  for path in args.train:
    replay_motions = json.loads((path / "eval.json").read_text())["motions"]
    if replay_motions != [motion.name for motion in motions]:
      p.error(f"Reference order in {path} differs from actor training motions")
  dev = "cuda:0"
  task = PushRollout(args.checkpoint, motions, 2, 2.0, 0.30, args.seed, dev)
  try:
    torch.manual_seed(args.seed)
    data = load_runs(args.train, dev)
    dynamics = ContactDynamics().to(dev)
    dynamics.load_state_dict(
      torch.load(args.dynamics, map_location=dev, weights_only=True)
    )
    dynamics.eval().requires_grad_(False)
    state = torch.load(
      args.checkpoint / "critic.pt", map_location=dev, weights_only=True
    )
    critic = build_mlp(state["0.weight"].shape[1], (256, 128), 1).to(dev)
    critic.load_state_dict(state)
    critic.eval().requires_grad_(False)
    policy = PushPolicy().to(dev)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0001)
    args.out.mkdir(parents=True, exist_ok=True)
    for it in range(1, args.updates + 1):
      stats = update(
        task, dynamics, policy, critic, data, optim, args.horizon, args.batch
      )
      if it % 20 == 0 or it == args.updates or it == 1:
        out = args.out / f"actor_residual_{it}.pt"
        torch.save(policy.state_dict(), out)
        print(it, json.dumps(stats), flush=True)
    (args.out / "run.json").write_text(
      json.dumps(
        {
          k: str(v) if isinstance(v, Path) else v
          for k, v in vars(args).items()
          if k != "train"
        }
        | {"train": [str(x) for x in args.train]},
        indent=2,
      )
      + "\n"
    )
  finally:
    task.env.close()


if __name__ == "__main__":
  main()
