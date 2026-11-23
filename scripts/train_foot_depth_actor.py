"""Paired Actor-only BPTT on identical camera snapshots, references and frozen dynamics."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from eval_walk_terrain import checkpoint_config
from mjlab.tasks.registry import load_env_cfg
from tensordict import TensorDict

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.models import build_mlp
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.foot_depth_actor import (
  FootDepthFusion,
  FutureFeet,
  MeasuredPatch,
  file_hash,
)
from supermimic.tasks.walk_terrain import WalkTerrain


def digest(tensors):
  h = hashlib.sha256()
  for x in tensors:
    h.update(x.detach().contiguous().cpu().numpy().tobytes())
  return h.hexdigest()


def train(args):
  device = args.device
  torch.cuda.set_device(device)
  actor_path = args.checkpoint / "actor.pt"
  base_hash = file_hash(actor_path)
  banks = sorted(
    (torch.load(p, weights_only=True) for p in args.banks), key=lambda b: b["motion"]
  )
  if len({b["motion"] for b in banks}) != len(banks):
    raise ValueError("Use one bank per distinct training clip")
  for b in banks:
    if (
      file_hash(Path(b["motion"])) != b["clip_sha256"] or b["base_sha256"] != base_hash
    ):
      raise ValueError("Camera bank and motion or collector weights do not match")
  env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  env_cfg.events = {
    k: v
    for k, v in env_cfg.events.items()
    if k in ("foot_friction", "base_com", "encoder_bias")
  }
  env_cfg.seed = args.seed
  env_cfg.scene.spec_fn = WalkTerrain(0.05)
  env_cfg.commands["motion"].motion_file = [b["motion"] for b in banks]
  env = create_env(env_cfg, 1, device)
  try:
    cmd = env.unwrapped.command_manager.get_term("motion")
    if cmd.motion.motion_names != [Path(b["motion"]).stem for b in banks]:
      raise ValueError("Motion loader order differs from bank order")
    ends = cmd.motion.motion_ends.tolist()
    for start, end, b in zip([0, *ends[:-1]], ends, banks, strict=True):
      times = b["tensors"]["motion_time"]
      if end - start != b["frames"] or times.min() < 0 or times.max() >= b["frames"]:
        raise ValueError("Recorded frame outside its motion clip")
      b["tensors"]["motion_time"] = times + start
    dataset = {
      k: torch.cat([b["tensors"][k] for b in banks]).to(device)
      for k in banks[0]["tensors"]
    }
    cfg = checkpoint_config(args.checkpoint)
    wm = build_world_model(cfg, env, device)
    wm.load_bundle(
      torch.load(
        args.checkpoint / "world_model.pt", map_location=device, weights_only=True
      )
    )
    wm.dynamics.eval().requires_grad_(False)
    actor = build_actor(cfg, wm, device)
    actor.load_state_dict(
      torch.load(actor_path, map_location=device, weights_only=True)
    )
    actor.eval().requires_grad_(False)
    critic_weights = torch.load(
      args.checkpoint / "critic.pt", map_location=device, weights_only=True
    )
    critic = build_mlp(
      critic_weights["0.weight"].shape[1], cfg.critic_hidden_dims, 1
    ).to(device)
    critic.load_state_dict(critic_weights)
    critic.eval().requires_grad_(False)
    if (
      cfg.terrain_inputs != "actor-world"
      or cfg.actor_obs != "rep_state"
      or cfg.history_length != 1
    ):
      raise ValueError("Expected the shared terrain-conditioned state Actor checkpoint")
    patch = MeasuredPatch(wm.terrain_query.offsets)
    wm.dynamics.terrain_query = patch
    imagined = MultiTaskWorldModelEnv(
      wm.dynamics,
      wm.representation,
      env,
      args.batch,
      None,
      actor.task_specs,
      1,
      cfg.termination_softness,
      device,
      terrain_query=patch,
      actor_sees_terrain=True,
      use_root_waypoints=cfg.use_root_waypoints,
    )
    feet = FutureFeet(cmd)
    names = list(cmd.motion.fk.joint_names)
    rollout_indices = torch.randint(
      len(dataset["state"]),
      (args.updates, args.batch),
      generator=torch.Generator().manual_seed(args.seed),
    )
    manifest = dict(
      seed=args.seed,
      updates=args.updates,
      batch=args.batch,
      horizon=args.horizon,
      learning_rate=args.lr,
      optimizer="Adam",
      gamma=cfg.policy.gamma,
      train_clips=[b["motion"] for b in banks],
      states=len(dataset["state"]),
      bank_sha256={str(p): file_hash(p) for p in args.banks},
      base_sha256=base_hash,
      critic_sha256=file_hash(args.checkpoint / "critic.pt"),
      world_model_sha256=file_hash(args.checkpoint / "world_model.pt"),
      schedule_sha256=digest([rollout_indices]),
      external_joint_compensation=False,
      device=torch.cuda.get_device_name(device),
      results={},
    )
    initial_return = None
    for mode in ("concat", "query"):
      torch.manual_seed(args.seed)
      head = FootDepthFusion(mode, names).to(device)
      initial = digest(head.parameters())
      actor.foot_depth_fusion = head
      optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
      history = []
      for it, indices in enumerate(rollout_indices.to(device)):
        torch.manual_seed(args.seed + it)
        batch = {k: v[indices] for k, v in dataset.items()}
        patch.bind(batch.pop("image"), batch.pop("origin"))
        obs = imagined.reset(TensorDict(batch, batch_size=[args.batch]))
        survival = 1.0
        total = 0.0
        gamma = cfg.policy.gamma
        for t in range(args.horizon):
          obs["depth_tokens"] = patch.tokens(imagined.aux_state)
          obs["future_feet"] = feet(imagined.motion_times)
          actions = actor(obs)
          if it == 0 and t == 0:
            with torch.no_grad():
              raw = actor.decode_actions(obs)
              torch.testing.assert_close(actions, raw, atol=0, rtol=0)
              visible = float(obs["depth_tokens"][..., 3].mean())
              if not 0 < visible <= 1:
                raise ValueError("Empty first camera batch")
          obs, reward, termination, _ = imagined.step(actions)
          total = total + gamma**t * survival * reward
          survival = survival * (1 - termination)
        bootstrap = (
          critic(obs["critic"])
          .clamp(max=imagined.max_step_reward / (1 - gamma))
          .squeeze(-1)
        )
        objective = (total + gamma**args.horizon * survival * bootstrap).mean()
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        pose_grad = head.pose[0].weight.grad.norm().item()
        depth_grad = head.depth[0].weight.grad.norm().item()
        norm = torch.nn.utils.clip_grad_norm_(
          head.parameters(), 1.0, error_if_nonfinite=True
        )
        if (
          not torch.isfinite(objective)
          or norm <= 0
          or (it == 1 and min(pose_grad, depth_grad) <= 0)
        ):
          raise RuntimeError(
            f"Invalid {mode} update {it}: {objective}, {norm}, {pose_grad}, {depth_grad}"
          )
        optimizer.step()
        history.append(
          dict(
            update=it + 1,
            objective=float(objective.detach()),
            grad_norm=float(norm),
            pose_grad=pose_grad,
            depth_grad=depth_grad,
          )
        )
        if it == 0 or (it + 1) % 16 == 0 or it + 1 == args.updates:
          print(mode, history[-1], flush=True)
      if initial_return is None:
        initial_return = history[0]["objective"]
        initial_sha = initial
      elif (
        initial_sha != initial or abs(initial_return - history[0]["objective"]) > 1.0e-6
      ):
        raise ValueError("The paired arms did not start identically")
      args.out.mkdir(parents=True, exist_ok=True)
      torch.save(
        dict(
          mode=mode,
          joint_names=names,
          base_sha256=base_hash,
          state_dict=head.state_dict(),
        ),
        args.out / f"{mode}.pt",
      )
      manifest["results"][mode] = dict(
        initial_sha256=initial,
        parameters=sum(p.numel() for p in head.parameters()),
        initial_visible=visible,
        updates=history,
      )
    manifest["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    (args.out / "training.json").write_text(
      json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
  finally:
    env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("out", type=Path)
  parser.add_argument("banks", nargs="+", type=Path)
  parser.add_argument("--updates", type=int, default=128)
  parser.add_argument("--batch", type=int, default=32)
  parser.add_argument("--horizon", type=int, default=8)
  parser.add_argument("--lr", type=float, default=0.0003)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if min(args.horizon, args.batch) < 1 or args.updates < 2:
    parser.error("Positive horizon and batch, and at least two updates are required")
  train(args)
