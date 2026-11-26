"""Distill adapted-reference actions and evaluate a camera-only residual controller."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from eval_scene_flat_regression import evaluate
from mjlab.tasks.tracking.mdp.commands import MotionLoader

from supermimic.tasks.foot_depth_actor import FutureFeet
from supermimic.tasks.online_terrain_adapter import OnlineTerrainAdapter, adapter_inputs


def digest(path):
  with Path(path).open("rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def paired_protocol(protocol, generated, names=None):
  manifest = json.loads((generated / "manifest.json").read_text())
  selected, raw_paths = [], []
  for clip in protocol["clips"]:
    raw = Path(clip["path"])
    if names is not None and raw.stem not in names:
      continue
    record = manifest["motion_clips"][raw.stem]
    adapted = generated / raw.name
    if (record["frames"] != clip["frames"]
        or record["source_sha256"] != clip["sha256"]
        or digest(raw) != clip["sha256"]
        or digest(adapted) != record["adapted_sha256"]):
      raise ValueError(f"Raw and adapted motion do not match the protocol: {raw.stem}")
    selected.append(dict(clip, path=str(adapted.resolve()), sha256=record["adapted_sha256"]))
    raw_paths.append(str(raw))
  if not selected or (names is not None and set(names) != {Path(p).stem for p in raw_paths}):
    raise ValueError("Requested clips are missing from the paired protocol")
  seed, scale = int(manifest["terrain_seed"]), float(manifest["rough_scale"])
  if not math.isfinite(scale) or scale <= 0:
    raise ValueError("Terrain scale must be finite and positive")
  return dict(protocol, clips=selected), raw_paths, seed, scale


class PairedActions:
  def __init__(self, paths, mode, adapter=None, stride=2):
    self.paths = paths
    self.mode = mode
    self.adapter = adapter
    self.stride = stride
    self.samples = []

  def bind(self, sim, motion, runner):
    if runner.use_root_waypoints or not runner.pelvis_motion_goal or runner.rough_reference != "off":
      raise ValueError("Expected the fixed 30K Actor observation layout")
    self.raw = MotionLoader(self.paths, motion.motion.fk, device=sim.device)
    if (self.raw.motion_names != motion.motion.motion_names
        or not torch.equal(self.raw.motion_ends, motion.motion.motion_ends)):
      raise ValueError("Raw and adapted motion indices must match")
    self.raw_feet = FutureFeet(SimpleNamespace(motion=self.raw, cfg=motion.cfg))
    manager = sim.action_manager
    if len(manager.active_terms) != 1:
      raise ValueError("Expected one G1 joint position action term")
    term = manager.get_term(manager.active_terms[0])
    self.scale = torch.as_tensor(term.scale, device=sim.device)
    self.offset = torch.as_tensor(term.offset, device=sim.device)
    if term.action_dim != 29 or not (self.scale > 0).all():
      raise ValueError("Expected 29 joint targets with positive action scales")
    self.limits = term._clip if term.cfg.clip is not None else None

  def act(self, actor, obs, representation, frames, aux, active, step):
    if self.mode == "teacher":
      return actor(obs)
    raw_obs = obs.clone(recurse=False)
    raw_obs["goal_motion"] = representation.compute_wm_goal(self.raw, frames, aux)[1]
    raw_obs["future_feet"] = self.raw_feet(frames)
    base_action = actor(raw_obs)
    if self.mode == "raw":
      return base_action
    if self.mode == "masked":
      # Mask only the new branch, preserving the pretrained Actor's input.
      raw_obs["depth_tokens"] = torch.zeros_like(raw_obs["depth_tokens"])
    features, visible = adapter_inputs(raw_obs, base_action)
    correction = (self.adapter(features, visible) / self.scale
                  if self.adapter is not None else 0.0)
    if self.mode == "collect" and step % self.stride == 0:
      teacher_action = actor(obs)
      base_target = base_action * self.scale + self.offset
      teacher_target = teacher_action * self.scale + self.offset
      if self.limits is None:
        lower, upper = torch.full_like(base_target, -torch.inf), torch.full_like(base_target, torch.inf)
      else:
        lower, upper = (limit.expand_as(base_target) for limit in self.limits.unbind(-1))
        teacher_target = teacher_target.clamp(lower, upper)
      self.samples.append(tuple(x[active].cpu() for x in (
        features, visible, base_target, teacher_target, lower, upper,
      )))
    return base_action + correction


def fit(model, optimizer, datasets, epochs, device):
  features, visible, base, target, lower, upper = (
    torch.cat(items) for items in zip(*datasets, strict=True)
  )
  if not visible.any():
    raise RuntimeError("The training rollouts contain no visible camera terrain")
  # Gather minibatches on the host so increasing the collection budget does not
  # increase GPU residency. Keep the aggregated earlier rollouts for DAgger.
  model.train()
  for _ in range(epochs):
    total = torch.zeros((), device=device)
    for index in torch.randperm(len(features)).split(1024):
      x, mask, q_base, q_target, lo, hi = (
        tensor[index].to(device) for tensor in (features, visible, base, target, lower, upper)
      )
      correction = model(x, mask)
      predicted = (q_base + correction).clamp(lo, hi)
      loss = (predicted - q_target).square().mean() + 0.01 * correction.square().mean()
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      total += loss.detach() * len(index)
    print(f"Distillation MSE + penalty: {total.item() / len(features):.8f}", flush=True)
  model.eval()


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("checkpoint", "protocol", "generated", "output"):
    parser.add_argument(name, type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--motion", nargs="+")
  parser.add_argument("--starts", type=Path, help="Reuse a matched snapshot for every arm")
  parser.add_argument("--steps", type=int)
  parser.add_argument("--rounds", type=int, default=0, help="Collect and fit this many rounds before evaluating")
  parser.add_argument("--train-terrain", action="append", type=Path, default=[],
                      help="Additional generated terrain to alternate between collection rounds")
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--stride", type=int, default=2)
  parser.add_argument("--max-delta-rad", type=float, default=0.12)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--weights", type=Path, help="Evaluate a saved student on a different terrain")
  parser.add_argument("--train-only", action="store_true")
  args = parser.parse_args()
  if bool(args.rounds) == bool(args.weights):
    parser.error("Specify either --rounds for training or --weights for evaluation")
  if (args.rounds < 0 or min(args.epochs, args.stride) < 1
      or (args.steps is not None and args.steps < 1)
      or not math.isfinite(args.max_delta_rad) or args.max_delta_rad <= 0):
    parser.error("Steps, stride, epochs, and maximum residual must be positive")
  if args.train_only and not args.rounds:
    parser.error("--train-only requires --rounds")
  if args.train_terrain and (args.rounds < len(args.train_terrain) + 1 or args.starts):
    parser.error("Training terrains require at least one round each and independent starts")
  torch.manual_seed(args.seed)
  original = json.loads(args.protocol.read_text())
  terrains = [paired_protocol(original, path, args.motion)
              for path in (args.generated, *args.train_terrain)]
  args.output.mkdir(parents=True, exist_ok=False)
  actor_hash = digest(args.checkpoint / "actor.pt")
  def json_path(value):
    return str(value) if isinstance(value, Path) else value

  config = {key: json_path(value) for key, value in vars(args).items()}
  config["train_terrain"] = [str(path) for path in args.train_terrain]
  manifest = dict(
    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    actor_sha256=actor_hash, protocol_sha256=digest(args.protocol),
    clips=[Path(path).stem for path in terrains[0][1]],
    terrains=[dict(seed=seed, scale=scale, manifest_sha256=digest(path / "manifest.json"))
              for path, (_, _, seed, scale) in zip((args.generated, *args.train_terrain), terrains, strict=True)],
    config=config,
  )
  write_json(args.output / "run_manifest.json", manifest)

  def replay(mode, adapter=None, terrain_index=0):
    protocol, paths, seed, scale = terrains[terrain_index]
    snapshot = "starts.npz" if terrain_index == 0 else f"starts-terrain-{terrain_index + 1}.npz"
    starts = args.starts or args.output / snapshot
    controller = PairedActions(paths, mode, adapter, args.stride)
    result = evaluate(
      args.checkpoint, protocol, starts, args.device,
      rough_seed=seed, rough_scale=scale, save_starts=not starts.exists(),
      max_steps=args.steps, direct_reference=True, action_controller=controller,
    )
    return result, controller.samples

  model = optimizer = None
  datasets = []
  for index in range(args.rounds):
    result, samples = replay("collect", model, index % len(terrains))
    write_json(args.output / f"collection-{index + 1}.json", result)
    datasets.extend(samples)
    if model is None:
      input_dim = samples[0][0].shape[-1]
      model = OnlineTerrainAdapter(input_dim, 29, args.max_delta_rad).to(args.device)
      optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"Round {index + 1}: {sum(len(batch[0]) for batch in datasets)} samples", flush=True)
    fit(model, optimizer, datasets, args.epochs, args.device)
    torch.save(dict(
      adapter=model.state_dict(), input_dim=input_dim, max_delta_rad=args.max_delta_rad,
      actor_sha256=actor_hash, train_manifest=manifest, round=index + 1,
    ), args.output / f"student-{index + 1}.pt")
  if args.train_only:
    return
  if args.weights:
    bundle = torch.load(args.weights, map_location=args.device, weights_only=True)
    if bundle["actor_sha256"] != actor_hash:
      raise ValueError("Student must be paired with its original tracking policy")
    model = OnlineTerrainAdapter(bundle["input_dim"], 29, bundle["max_delta_rad"]).to(args.device)
    model.load_state_dict(bundle["adapter"])
    model.eval()
  for arm in ("teacher", "raw", "student", "masked"):
    result, _ = replay(arm, model)
    write_json(args.output / f"{arm}.json", result)
    print(arm, result["completed"], result["mean_pelvis_xy_error_m"], flush=True)


if __name__ == "__main__":
  main()
