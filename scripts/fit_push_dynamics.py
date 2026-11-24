"""Fit box motion and robot reaction from first-episode G1 simulator transitions."""

import argparse
import json
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.models_push import ContactDynamics
from supermimic.tasks.push_rollout import PushRollout


def load_runs(paths, device):
  stacks = []
  for path in paths:
    data = torch.load(path / "transitions.pt", map_location=device, weights_only=True)
    meta = json.loads((path / "eval.json").read_text())
    data["physics"] = torch.tensor(
      [meta["mass"] / 3.0, meta["friction"]], device=device
    ).expand(*data["valid"].shape, 2)
    stacks.append(data)
  keys = (
    "action",
    "full",
    "after_full",
    "box",
    "after_box",
    "palms",
    "after_palms",
    "contact",
    "valid",
    "ctx",
    "frame",
    "goal",
    "physics",
  )
  return {k: torch.cat([data[k] for data in stacks], dim=1) for k in keys}


def sample_paths(data, horizon, n):
  if horizon < 1 or horizon > len(data["valid"]):
    raise ValueError("Requested training window does not fit in recorded rollout")
  t, b = data["valid"][: 1 - horizon if horizon > 1 else None].nonzero(as_tuple=True)
  # alive is monotonically false after the first reset.
  ok = data["valid"][t + horizon - 1, b]
  valid_ids = torch.where(ok)[0]
  if not len(valid_ids):
    raise ValueError("No uninterrupted simulator episodes span this training horizon")
  take = valid_ids[torch.randint(len(valid_ids), (n,), device=b.device)]
  return t[take], b[take]


@torch.no_grad()
def prepare_robot(task, data):
  form, dyn = task.wm.representation, task.wm.dynamics
  result = {}
  for t in range(data["full"].shape[0]):
    full, after = data["full"][t], data["after_full"][t]
    rep, aux = form.represent(full), form.extract_aux_state(full)
    base = dyn.predict_delta(rep, data["action"][t], aux, data["ctx"][t])
    next_aux, next_rep = form.apply_prediction(aux, rep, base, dyn.dt)
    pos, rot, _, _ = form.world_frame_bodies(next_aux, next_rep)
    target = torch.cat(
      [
        quat_apply_inverse(aux[:, 3:7], after[:, 36:39] - full[:, 36:39]),
        quat_apply_inverse(aux[:, 3:7], after[:, 39:42] - full[:, 39:42]),
        after[:, 42:] - full[:, 42:],
      ],
      -1,
    )
    for name, val in [
      ("robot_target", target - base),
      ("normalized_state", dyn.state_normalizer(rep)),
      ("nominal_palms", task.palms(pos, rot)),
    ]:
      result.setdefault(name, []).append(val)
  data.update({k: torch.stack(v) for k, v in result.items()})


def fit(model, task, data, steps):
  prepare_robot(task, data)
  scale = data["robot_target"][data["valid"]].std(0).clamp_min(0.05)
  optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0e-5)
  h = 6
  for it in range(steps):
    t, b = sample_paths(data, h, 256)
    box = data["box"][t, b]
    total = 0.0
    for j in range(h):
      i = t + j
      box = model.advance_box(
        box, data["palms"][i, b], data["after_palms"][i, b], data["physics"][i, b]
      )
      real = data["after_box"][i, b]
      # Use quaternion sign-invariant orientation error and multi-step position/velocity loss.
      orientation = 1.0 - (box[:, 3:7] * real[:, 3:7]).sum(-1).square().clamp(max=1.0)
      total = (
        total
        + ((box[:, :3] - real[:, :3]) / 0.04).square().mean()
        + orientation.mean() * 4
      )
      total = total + ((box[:, 7:] - real[:, 7:]) / model.twist_scale).square().mean()
    correction = model.robot_delta(
      data["normalized_state"][t, b],
      data["action"][t, b],
      data["box"][t, b],
      data["palms"][t, b],
      data["nominal_palms"][t, b],
      data["physics"][t, b],
    )
    rloss = ((correction - data["robot_target"][t, b]) / scale).square().mean()
    loss = total / h + rloss
    if not torch.isfinite(loss):
      raise ValueError(f"Nonfinite dynamics loss at update {it}")
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optim.step()
    if it % 100 == 0 or it + 1 == steps:
      print(
        f"fit {it + 1} box_loss {total.item() / h:.5f} robot_loss {rloss.item():.5f}",
        flush=True,
      )


def fit_coupled(model, task, data, steps):
  """Use recorded actions throughout each window; the two model states roll forward together."""
  form = task.wm.representation
  optim = torch.optim.AdamW(model.parameters(), lr=0.00012, weight_decay=1.0e-5)
  for it in range(steps):
    t, b = sample_paths(data, 8, 64)
    initial = data["full"][t, b]
    aux, rep = form.extract_aux_state(initial), form.represent(initial)
    box, palms = data["box"][t, b], data["palms"][t, b]
    loss = 0.0
    for step in range(8):
      i = t + step
      aux, rep, box, palms = model.step(
        task,
        aux,
        rep,
        data["action"][i, b],
        data["ctx"][i, b],
        box,
        palms,
        data["physics"][i, b],
      )
      actual = data["after_full"][i, b]
      actual_rep = form.represent(actual)
      obj = data["after_box"][i, b]
      loss += ((palms - data["after_palms"][i, b]) / 0.04).square().mean()
      loss += ((aux[:, :3] - actual[:, :3]) / 0.035).square().mean()
      loss += (
        0.25 * (rep[:, 4:10] - actual_rep[:, 4:10]).square().mean()
        + 0.015 * (rep[:, 39:] - actual_rep[:, 39:]).square().mean()
      )
      loss += ((box[:, :3] - obj[:, :3]) / 0.04).square().mean() + 0.4 * (
        (box[:, 7:] - obj[:, 7:]) / model.twist_scale
      ).square().mean()
    loss /= 8
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optim.step()
    if it % 40 == 0 or it + 1 == steps:
      print(f"coupled update {it + 1}, loss {loss.item():.5f}", flush=True)


@torch.no_grad()
def evaluate(model, task, data, seed):
  torch.manual_seed(seed)
  form = task.wm.representation
  t, b = sample_paths(data, 32, 512)
  start = data["box"][t, b]
  box_pred, pred_oracle = start, start
  aux, rep = (
    form.extract_aux_state(data["full"][t, b]),
    form.represent(data["full"][t, b]),
  )
  palms = data["palms"][t, b]
  any_contact = torch.zeros_like(t, dtype=torch.bool)
  out = {}
  for step in range(32):
    i = t + step
    any_contact |= data["contact"][i, b].any(-1)
    aux, rep, box_pred, palms = model.step(
      task,
      aux,
      rep,
      data["action"][i, b],
      data["ctx"][i, b],
      box_pred,
      palms,
      data["physics"][i, b],
    )
    pred_oracle = model.advance_box(
      pred_oracle, data["palms"][i, b], data["after_palms"][i, b], data["physics"][i, b]
    )
    if step + 1 not in (1, 8, 32):
      continue
    real = data["after_box"][i, b, :3]
    estimates = {
      "coupled_cm": box_pred[:, :3],
      "measured_palm_cm": pred_oracle[:, :3],
      "static_cm": start[:, :3],
      "constant_velocity_cm": start[:, :3] + (step + 1) * model.dt * start[:, 7:10],
    }
    out[step + 1] = {}
    for group, mask in (
      ("all", torch.ones_like(any_contact)),
      ("contact", any_contact),
      ("no_contact", ~any_contact),
    ):
      if not mask.any():
        continue
      out[step + 1][group] = {
        "n": int(mask.sum()),
        **{
          k: float((v - real)[mask].norm(dim=-1).mean() * 100)
          for k, v in estimates.items()
        },
      }
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("checkpoint", type=Path)
  p.add_argument("motions", type=Path)
  p.add_argument("out", type=Path)
  p.add_argument("--train", nargs="+", type=Path, required=True)
  p.add_argument("--test", nargs="+", type=Path, required=True)
  p.add_argument("--updates", type=int, default=600)
  p.add_argument("--joint-updates", type=int, default=0)
  p.add_argument("--warmstart", type=Path)
  p.add_argument("--seed", type=int, default=42)
  args = p.parse_args()
  dev = "cuda:0"
  torch.manual_seed(args.seed)
  task = PushRollout(
    args.checkpoint, sorted(args.motions.glob("*.npz")), 2, 2.0, 0.3, args.seed, dev
  )
  try:
    train = load_runs(args.train, dev)
    model = ContactDynamics().to(dev)
    if args.warmstart:
      model.load_state_dict(
        torch.load(args.warmstart, map_location=dev, weights_only=True)
      )
    fit(model, task, train, args.updates)
    fit_coupled(model, task, train, args.joint_updates)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "contact_dynamics.pt")
    val = {
      str(path): evaluate(model, task, load_runs([path], dev), 71) for path in args.test
    }
    result = {
      "training_transitions": int(train["valid"].sum()),
      "updates": args.updates,
      "joint_updates": args.joint_updates,
      "validation": val,
    }
    (args.out / "dynamics_eval.json").write_text(
      json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
  finally:
    task.env.close()


if __name__ == "__main__":
  main()
