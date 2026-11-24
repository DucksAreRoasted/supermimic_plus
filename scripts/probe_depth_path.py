"""Perturb only global reference-to-pelvis XY on real replay states; measure Actor observability."""

import argparse
import json
from pathlib import Path

import torch
from eval_walk_terrain import checkpoint_config
from mjlab.tasks.registry import load_env_cfg

from supermimic.representations import root_waypoints
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.foot_depth_actor import FutureFeet, MeasuredPatch, file_hash
from supermimic.tasks.task_spec import build_actor_obs_td


@torch.inference_mode()
def probe(checkpoint, bank_path, device):
  bank = torch.load(bank_path, weights_only=True)
  motion = Path(bank["motion"])
  if not motion.is_file():
    motion = (
      Path(__file__).resolve().parents[1]
      / "data/modelscope/LAFAN1_Retargeting_Dataset/npz"
      / motion.name
    )
  if file_hash(motion) != bank["clip_sha256"]:
    raise ValueError("Recorded reference does not match")
  cfg = checkpoint_config(checkpoint)
  env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  env_cfg.commands["motion"].motion_file = str(motion)
  env_cfg.events = {
    k: v
    for k, v in env_cfg.events.items()
    if k in ("foot_friction", "base_com", "encoder_bias")
  }
  cfg.configure_terrain(env_cfg)
  env = create_env(env_cfg, 1, device)
  try:
    wm = build_world_model(cfg, env, device)
    actor = build_actor(cfg, wm, device)
    actor.load_state_dict(
      torch.load(checkpoint / "actor.pt", map_location=device, weights_only=True)
    )
    actor.eval()
    command = env.unwrapped.command_manager.get_term("motion")
    rows = {k: v[::9].to(device) for k, v in bank["tensors"].items()}
    time = rows["motion_time"]
    rep, aux = (
      wm.representation.represent(rows["state"]),
      wm.representation.extract_aux_state(rows["state"]),
    )
    patch = MeasuredPatch(wm.terrain_query.offsets)
    patch.bind(rows["image"], rows["origin"])
    feet = FutureFeet(command)(time)
    full, goal = wm.representation.compute_wm_goal(command.motion, time, aux)
    obs = build_actor_obs_td({"motion": goal}, rep[:, None])
    obs["terrain"], obs["depth_tokens"], obs["future_feet"] = (
      patch.features(aux),
      patch.tokens(aux),
      feet,
    )
    waypoints = root_waypoints(full)
    obs["root_waypoints"] = waypoints
    initial = actor(obs)
    changes = []
    for dx, dy in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
      displaced = aux.clone()
      displaced[:, :2] += aux.new_tensor([dx, dy])
      full_b, goal_b = wm.representation.compute_wm_goal(
        command.motion, time, displaced
      )
      obs["goal_motion"], obs["root_waypoints"] = goal_b, root_waypoints(full_b)
      delta = (actor(obs) - initial).abs()
      changes.append(
        dict(
          dx=dx,
          dy=dy,
          goal_max_abs=float((goal_b - goal).abs().max()),
          waypoint_mean_abs=float((obs["root_waypoints"] - waypoints).abs().mean()),
          action_mean_abs=float(delta.mean()),
          action_max_abs=float(delta.max()),
        )
      )
    return dict(
      actor_sha256=file_hash(checkpoint / "actor.pt"),
      states=len(time),
      use_root_waypoints=cfg.use_root_waypoints,
      offsets=changes,
    )
  finally:
    env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("bank", type=Path)
  parser.add_argument("out", type=Path)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  result = probe(args.checkpoint, args.bank, args.device)
  args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(json.dumps(result, indent=2))
