"""Fine-tune only the online terrain residual with on-policy MuJoCo rollouts."""

import argparse
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import torch
from eval_walk_terrain import checkpoint_config
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from run_g1_online_terrain import digest, write_json
from tensordict import TensorDict

from supermimic.representations import GenCoordRepresentation
from supermimic.runners.supermimic_runner import build_actor
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.depth_mimic import DepthMemory, walking_camera
from supermimic.tasks.foot_depth_actor import FutureFeet, MeasuredPatch
from supermimic.tasks.online_terrain_adapter import OnlineTerrainActions
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.terrain_reset import TerrainResetMotionCommandCfg
from supermimic.tasks.walk_terrain import IrregularWalkTerrain, TerrainObservation

CONTACTS = ("online_sole_field", "online_sole_plane")


def contact_config(name, geom):
  return ContactSensorCfg(
    name=name, primary=ContactMatch("subtree", ("left_ankle_roll_link", "right_ankle_roll_link"), "robot"),
    secondary=ContactMatch("geom", geom), fields=("found",), reduce="netforce", num_slots=1,
  )


class StanceSlip:
  def __init__(self, cfg, env):
    self.robot = env.scene["robot"]
    names = env.scene[CONTACTS[0]].primary_names
    feet = [name.split("/")[-1].replace("_ankle_roll_link", "_foot") for name in names]
    self.sites, found = self.robot.find_sites(feet, preserve_order=True)
    if len(self.sites) != 2 or list(found) != feet or env.scene[CONTACTS[1]].primary_names != names:
      raise ValueError("Expected two ordered G1 sole sites and matching contact sensors")

  def __call__(self, env):
    contact = (env.scene[CONTACTS[0]].data.found > 0) | (env.scene[CONTACTS[1]].data.found > 0)
    speed_sq = self.robot.data.site_lin_vel_w[:, self.sites, :2].square().sum(-1)
    return (speed_sq * contact).sum(-1)


def root_drift_cost(env):
  motion = env.command_manager.get_term("motion")
  reference_xy = motion.motion.root_pos_w[motion.time_steps, :2] + env.scene.env_origins[:, :2]
  xy = env.sim.data.qpos[:, :2]
  return ((xy - reference_xy).norm(dim=-1) / 0.15).square().clamp_max(9.0)


def failure_cost(env):
  return env.termination_manager.terminated.float()


class Observations:
  def __init__(self, sim, runner, camera_cfg, actor, representation, motion, policy, ground):
    self.sim = sim
    self.actor = actor
    self.motion = motion
    self.representation = representation
    self.policy = policy
    self.feet = FutureFeet(motion)
    self.camera = sim.scene[camera_cfg.name]
    self.memory = DepthMemory(sim.num_envs, sim.device, sim.step_dt, camera_cfg.pattern)
    self.patch = MeasuredPatch(ground.offsets)
    policy.bind(sim, motion, runner, self.patch)

  def __call__(self, obs):
    rep = self.representation.represent(obs["full_state"])
    aux = self.representation.extract_aux_state(obs["full_state"])
    frames = self.motion.time_steps
    _, goal = self.representation.compute_wm_goal(self.motion.motion, frames, aux)
    td = build_actor_obs_td({"motion": goal}, rep[:, None])
    self.camera.observe(self.memory, aux, terrain_only=False)
    self.patch.bind(*self.patch.capture(self.memory))
    td["terrain"], td["depth_tokens"] = self.patch.features_and_tokens(aux)
    td["future_feet"] = self.feet(frames)
    base, features, visible = self.policy.policy_inputs(self.actor, td, frames, aux)
    return TensorDict({"features": features}, batch_size=[len(rep)]), (base, visible, frames, aux)


def train(args, clips, manifest, runner, initial):
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  cfg.seed = args.seed
  cfg.events = {}
  cfg.episode_length_s = max(c["frames"] for c in clips) * 0.02 + 1.0
  cfg.scene.env_spacing = 0.0
  cfg.scene.spec_fn = IrregularWalkTerrain(
    height=0.09 * args.rough_scale, lateral_height=0.04 * args.rough_scale,
    landscape_seed=args.terrain_seed,
  )
  camera_cfg = walking_camera(runner.camera_profile)
  cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg,
                       contact_config(CONTACTS[0], "walking_surface"),
                       contact_config(CONTACTS[1], "terrain"))
  cfg.sim.mujoco.iterations = cfg.sim.mujoco.ls_iterations = 60
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1024)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 32)
  for group in cfg.observations.values():
    group.enable_corruption = False
  motion_cfg = TerrainResetMotionCommandCfg.from_motion(cfg.commands["motion"])
  motion_cfg.motion_file = [c["path"] for c in clips]
  motion_cfg.sampling_mode = "uniform"
  motion_cfg.pose_range, motion_cfg.velocity_range, motion_cfg.joint_position_range = {}, {}, (0.0, 0.0)
  if args.recovery_reset:
    motion_cfg.pose_range = {"x": (-0.035, 0.035), "y": (-0.035, 0.035),
                             "roll": (-0.12, 0.12), "pitch": (-0.12, 0.12)}
    motion_cfg.velocity_range = {"x": (-0.3, 0.3), "y": (-0.3, 0.3),
                                 "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3)}
  cfg.commands["motion"] = motion_cfg
  cfg.rewards.update(
    online_stance_slip=RewardTermCfg(func=StanceSlip, weight=-args.slip_weight),
    online_root_drift=RewardTermCfg(func=root_drift_cost, weight=-args.drift_weight),
    online_failure=RewardTermCfg(func=failure_cost, weight=-args.failure_weight / 0.02),
  )
  env = create_env(cfg, args.envs, args.device)
  try:
    sim = env.unwrapped
    obs, _ = sim.reset(seed=args.seed)
    motion = sim.command_manager.get_term("motion")
    rep = GenCoordRepresentation(29, motion.robot.data.default_joint_pos[0])
    rep.actor_obs_mode, rep.pelvis_motion_goal, rep.fk = runner.actor_obs, runner.pelvis_motion_goal, motion.motion.fk
    world = SimpleNamespace(representation=rep, num_actions=29, num_goals=obs["goal"].shape[-1])
    actor = build_actor(runner, world, args.device)
    actor.load_state_dict(torch.load(args.checkpoint / "actor.pt", map_location=args.device, weights_only=True))
    actor.eval().requires_grad_(False)
    policy = OnlineTerrainActions(args.initial, args.checkpoint / "actor.pt", args.device,
                                  max_root_drift=args.max_root_drift)
    ground = TerrainObservation(cfg.scene.spec_fn).to(args.device)
    observe = Observations(sim, runner, camera_cfg, actor, rep, motion, policy, ground)
    with torch.no_grad():
      features, context = observe(obs)
    groups = {"actor": ["features"], "critic": ["features"]}
    residual = MLPModel(features, groups, "actor", 29, hidden_dims=(128,),
                        distribution_cfg={"class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                                          "init_std": args.noise_std}).to(args.device)
    residual.mlp = policy.model.net
    value = MLPModel(features, groups, "critic", 1, hidden_dims=(256, 128)).to(args.device)
    storage = RolloutStorage("rl", args.envs, args.steps, features, [29], device=args.device)
    alg = PPO(actor=residual, critic=value, storage=storage, device=args.device,
              num_learning_epochs=4, num_mini_batches=4, clip_param=0.2, gamma=0.99,
              lam=0.95, desired_kl=0.01, learning_rate=args.lr, max_grad_norm=1.0,
              schedule="fixed")
    metrics_path = args.output / "train.jsonl"
    for iteration in range(1, args.iterations + 1):
      alg.eval_mode()
      with torch.no_grad():
        totals = torch.zeros(5, device=args.device)
        for _ in range(args.steps):
          base, visible, frames, aux = context
          logits = alg.act(features)
          delta = policy.model.decode(logits, visible)
          action = policy.apply_residual(base, delta, frames, aux)
          obs, reward, terminated, truncated, _ = sim.step(action)
          dones = terminated | truncated
          observe.memory.reset(dones)
          features, context = observe(obs)
          alg.process_env_step(features, reward, dones, {})
          contact = (sim.scene[CONTACTS[0]].data.found > 0) | (sim.scene[CONTACTS[1]].data.found > 0)
          totals += torch.stack((reward.mean(), terminated.float().mean(),
                                 visible.float().mean(), delta.abs().mean(), contact.float().mean()))
        alg.compute_returns(features)
      alg.train_mode()
      losses = alg.update()
      row = dict(zip(("reward", "failure_per_step", "visible", "mean_delta_rad", "contact"),
                     (totals / args.steps).tolist(), strict=True))
      row.update(iteration=iteration, **losses)
      with metrics_path.open("a") as out:
        out.write(json.dumps(row, allow_nan=False) + "\n")
      if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
        print(json.dumps(row), flush=True)
      if iteration % args.save_interval == 0 or iteration == args.iterations:
        torch.save(dict(adapter=policy.model.state_dict(), input_dim=initial["input_dim"],
                        max_delta_rad=initial["max_delta_rad"], actor_sha256=initial["actor_sha256"],
                        foot_context=policy.foot_context, train_manifest=manifest, round=iteration),
                   args.output / f"student-ppo-{iteration}.pt")
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("checkpoint", "initial", "protocol", "output"):
    parser.add_argument(name, type=Path)
  for name, default in (("envs", 32), ("steps", 32), ("iterations", 80), ("save-interval", 20),
                        ("terrain-seed", 1714), ("seed", 42)):
    parser.add_argument(f"--{name}", type=int, default=default)
  for name, default in (("rough-scale", 0.8), ("lr", 2e-5), ("noise-std", 0.06),
                        ("slip-weight", 0.5), ("drift-weight", 1.0), ("failure-weight", 1.0),
                        ("max-root-drift", 0.18)):
    parser.add_argument(f"--{name}", type=float, default=default)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--recovery-reset", action="store_true",
                      help="Perturb planar pelvis pose and velocity at training resets")
  args = parser.parse_args()
  if min(args.envs, args.steps, args.iterations, args.save_interval) < 1 or args.envs * args.steps % 4:
    parser.error("Need positive counts and a rollout batch divisible by four")
  for name in ("rough_scale", "lr", "noise_std", "slip_weight", "drift_weight", "failure_weight", "max_root_drift"):
    value = getattr(args, name)
    if not math.isfinite(value) or value <= 0:
      parser.error(f"{name} must be finite and positive")
  clips = json.loads(args.protocol.read_text())["clips"]
  if not clips or any(digest(c["path"]) != c["sha256"] for c in clips):
    raise ValueError("Expected checksum-verified raw training clips")
  runner = checkpoint_config(args.checkpoint)
  if (runner.camera_profile != "instinct" or runner.depth_fusion == "none"
      or runner.actor_obs != "rep_state" or runner.history_length != 1
      or not runner.pelvis_motion_goal or runner.rough_reference != "off"
      or runner.use_root_waypoints or runner.imagined_vision != "none"):
    raise ValueError("Expected the 30K single-frame measured-depth tracking Actor")
  initial = torch.load(args.initial, map_location="cpu", weights_only=True)
  actor_hash = digest(args.checkpoint / "actor.pt")
  if actor_hash != initial["actor_sha256"]:
    raise ValueError("Student was trained against a different tracking Actor")
  args.output.mkdir(parents=True, exist_ok=False)
  manifest = dict(
    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    actor_sha256=actor_hash, initial_sha256=digest(args.initial), protocol_sha256=digest(args.protocol),
    config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
  )
  write_json(args.output / "run_manifest.json", manifest)
  train(args, clips, manifest, runner, initial)


if __name__ == "__main__":
  main()
