"""Online MuJoCo training of G1 on rough ground with two nonlinear dynamics heads."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.foot_depth_actor import file_hash

ROOT = Path(__file__).resolve().parents[1]
CLIPS = (
  "walk1_subject1",
  "walk1_subject2",
  "walk1_subject5",
  "walk2_subject4",
  "walk3_subject1",
  "walk3_subject3",
  "walk4_subject1",
)


ACTOR_INPUTS = {
  "proprio97x10": ("proprio_pelvis", 10),
  "proprio10": ("proprio", 10),
  "proprio1": ("proprio", 1),
  "state1": ("rep_state", 1),
}


def scratch_config(iterations, batch, seed, root_waypoints, num_envs, *, terrain_height_reset=False,
                   official_backends=False, cache_reference=False, wm_lr=2e-5, actor_input="proprio10",
                   pelvis_motion_goal=True):
  waypoint_label = "-waypoints" if root_waypoints else ""
  reset_label = "-terrainreset" if terrain_height_reset else ""
  speed_label = (f"-batch{batch}" if batch != 64 else "") + ("-tf32" if official_backends else "")
  if cache_reference:
    speed_label += "-refcache"
  if not math.isfinite(wm_lr) or wm_lr <= 0:
    raise ValueError("The world-model learning rate must be positive and finite")
  if wm_lr != 2e-5:
    speed_label += f"-wmlr{wm_lr:g}"
  actor_obs, history_length = ACTOR_INPUTS[actor_input]
  if actor_input != "state1":
    speed_label += f"-{actor_input}"
  if pelvis_motion_goal:
    speed_label += "-goal68"
  cfg = SuperMimicRunnerCfg(
    pelvis_motion_goal=pelvis_motion_goal,
    actor_obs=actor_obs,
    history_length=history_length,
    run_name=f"g1-rough7-concat-wm-dualmlp128-cal0{waypoint_label}{reset_label}{speed_label}-scratch-env{num_envs}-{iterations}it-seed{seed}",
    max_iterations=iterations,
    log_interval=4 if iterations < 100 else 64,
    save_interval=1000,
    residual_shared_trunk=True,
    residual_correction=True,
    shared_head_width=128,
    depth_fusion="concat",
    cache_motion_reference=cache_reference,
    terrain_inputs="actor-world",
    use_root_waypoints=root_waypoints,
    terrain_height=0.09,
    terrain_lateral_height=0.04,
    terrain_landscape_seed=701,
    terrain_height_reset=terrain_height_reset,
    critic_variant="sim-calibrated",
    sim_calibration_weight=0.0,
    replay_ratio=2,
    buffer_size=512,
  )
  cfg.return_sampler.focus_share = 0.4
  cfg.wm.lr = wm_lr
  cfg.policy.lr = 2e-5
  cfg.wm.num_batches = 1
  cfg.wm.batch_size = cfg.policy.batch_size = batch
  cfg.wm.horizon, cfg.policy.horizon = 8, 16
  cfg.policy.progress_lr_start = cfg.policy.progress_lr_end = 1.0
  return cfg


def initial_parameter_hash(module, *, omit_waypoint_columns=False, exclude_prefix=None):
  digest = hashlib.sha256()
  for name, param in module.named_parameters():
    if exclude_prefix and name.startswith(exclude_prefix):
      continue
    if omit_waypoint_columns and name == "decoder.0.weight":
      if not module.use_root_waypoints or param[:, -4:].count_nonzero().item():
        raise ValueError("Expected four zero-initialized waypoint columns")
      param = param[:, :-4]
    digest.update(name.encode() + b"\0")
    digest.update(param.detach().contiguous().cpu().numpy().tobytes())
  return digest.hexdigest()


def protocol_clips(path):
  clips = json.loads(path.read_text())["clips"]
  paths = [ROOT / c["path"] for c in clips]
  if len(paths) != 40 or len({p.stem for p in paths}) != 40:
    raise ValueError("Expected 40 distinct LAFAN motion clips")
  for p, entry in zip(paths, clips, strict=True):
    if file_hash(p) != entry["sha256"]:
      raise ValueError(f"LAFAN protocol checksum mismatch: {p}")
  return paths


def build_scratch_runner(cfg, num_envs, seed, *, clips=None):
  task = "SuperMimic-GenCoord-G1"
  env_cfg = load_env_cfg(task, play=False)
  env_cfg.seed = seed
  torch.manual_seed(seed)
  cfg.configure_terrain(env_cfg)
  env_cfg.sim.mujoco.iterations = env_cfg.sim.mujoco.ls_iterations = 60
  env_cfg.events = {
    k: v
    for k, v in env_cfg.events.items()
    if k in ("foot_friction", "base_com", "encoder_bias")
  }
  command = env_cfg.commands["motion"]
  if clips is None:
    clips = [
      ROOT / "data/modelscope/LAFAN1_Retargeting_Dataset/npz" / (name + ".npz")
      for name in CLIPS
    ]
  command.motion_file = [str(p) for p in clips]
  command.sampling_mode = "external"
  print(
    json.dumps(
      dict(
        clips={p.name: file_hash(p) for p in clips},
        initialization="model_transfer" if cfg.init_model_dir else "scratch",
        seed=seed,
      )
    ),
    flush=True,
  )
  env = create_env(env_cfg, num_envs, "cuda:0")
  return SuperMimicRunner(env, cfg, task, device="cuda:0")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--iterations", type=int, default=40000)
  parser.add_argument("--goal-width", type=int, choices=(64, 68),
                      help="Reference features per frame (scratch default: 68, transfer: source layout)")
  parser.add_argument("--actor-input", choices=tuple(ACTOR_INPUTS),
                      help="Default: proprio10 from scratch, or the source Actor layout for visual transfer")
  parser.add_argument("--root-waypoints", action="store_true",
                      help="Expose current and future reference pelvis XY in the Actor decoder")
  parser.add_argument("--terrain-curriculum", action="store_true",
                      help="Start flat, then ramp to the existing mild rough-ground benchmark")
  parser.add_argument("--hold-imagined-vision", action="store_true",
                      help="Train latent forecasts, but hold the initial feature during policy rollouts")
  parser.add_argument("--terrain-height-reset", action="store_true",
                      help="Add the local ground height to the sampled pelvis reset pose")
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--batch", type=int, default=64)
  parser.add_argument("--policy-batch", type=int, help="Override Actor rollout batch size independently of the world-model batch")
  parser.add_argument("--actor-microbatch", type=int, help="Accumulate the single Actor update over smaller rollout batches")
  parser.add_argument("--wm-lr", type=float, default=2e-5,
                      help="Set the world-model learning rate independently of the policy")
  parser.add_argument("--cache-reference", action="store_true",
                      help="Precompute FK and future ankle features for the fixed reference clips")
  parser.add_argument("--official-backends", action="store_true",
                      help="Use the same Torch backend and TF32 configuration as the official entry point")
  parser.add_argument("--vision-base-checkpoint", type=Path, help="Initialize the two physical heads, Actor and Critic from an existing checkpoint")
  parser.add_argument("--training-reference", type=Path,
                      help="Use replay and optimization settings from an archived run manifest")
  parser.add_argument("--flat-depth-arm", choices=("base", "aux", "full"),
                      help="Flat ground with leg fusion; compare no vision, auxiliary-only, or full latent")
  parser.add_argument("--max-iteration-compute-seconds", type=float, default=0.0)
  parser.add_argument("--joint-visual-dynamics", action="store_true")
  parser.add_argument("--vision-hidden-fusion", action="store_true")
  parser.add_argument("--detach-visual-state", action="store_true")
  parser.add_argument("--imagined-vision", choices=("none", "render", "latent"), default="none")
  parser.add_argument("--lafan40-protocol", type=Path, help="Use all 40 checksum-verified clips, in protocol order")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--expect-initial-hashes", type=Path,
                      help="Require these scratch parameter digests before collecting any data")
  parser.add_argument("--expect-control-hashes", type=Path,
                      help="Check shared scratch parameters against the run without root waypoints")
  args = parser.parse_args()
  if min(args.iterations, args.batch, args.num_envs) < 1:
    parser.error("iterations, batch and num-envs must be positive")
  if args.policy_batch is not None and args.policy_batch < 1:
    parser.error("--policy-batch must be positive")
  if args.official_backends:
    configure_torch_backends()
  actor_input = args.actor_input or "proprio10"
  goal_width = args.goal_width or 68
  if args.vision_base_checkpoint:
    source = json.loads((args.vision_base_checkpoint / "run_manifest.json").read_text())
    source_cfg = source["runner_cfg"]
    source_width = 68 if source_cfg.get("pelvis_motion_goal", False) else 64
    if args.goal_width is not None and args.goal_width != source_width:
      parser.error(f"Visual transfer requires source goal width {source_width}")
    goal_width = source_width
    layout = (source_cfg.get("actor_obs", "rep_state"), source_cfg.get("history_length", 1))
    if args.actor_input is None:
      actor_input = next((name for name, dims in ACTOR_INPUTS.items() if dims == layout), None)
    if actor_input is None or ACTOR_INPUTS[actor_input] != layout:
      parser.error(f"Visual transfer requires the source Actor layout {layout}")
  cfg = scratch_config(args.iterations, args.batch, args.seed, args.root_waypoints, args.num_envs,
                       terrain_height_reset=args.terrain_height_reset, official_backends=args.official_backends,
                       cache_reference=args.cache_reference, wm_lr=args.wm_lr, actor_input=actor_input,
                       pelvis_motion_goal=goal_width == 68)
  if args.terrain_curriculum:
    if (args.iterations < 20 or args.flat_depth_arm or args.vision_base_checkpoint
        or not args.terrain_height_reset or args.imagined_vision == "render"):
      parser.error("Terrain curriculum requires a scratch run of at least 20 iterations with terrain-height resets")
    cfg.terrain_height *= 0.25
    cfg.terrain_lateral_height *= 0.25
    cfg.terrain_landscape_seed = 1701
    cfg.terrain_curriculum = tuple(
      (args.iterations * percent // 100, scale)
      for percent, scale in ((0, 0.0), (20, 0.25), (30, 0.5), (40, 0.75), (50, 1.0))
    )
    cfg.run_name += "-curriculum"
  cfg.max_iteration_compute_seconds = args.max_iteration_compute_seconds
  cfg.joint_visual_dynamics = args.joint_visual_dynamics
  if cfg.joint_visual_dynamics:
    cfg.run_name += "-jointwm"
  cfg.vision_hidden_fusion = args.vision_hidden_fusion
  cfg.wm.detach_visual_state = args.detach_visual_state
  if cfg.vision_hidden_fusion:
    cfg.run_name += "-hiddenfusion"
  cfg.hold_imagined_vision = args.hold_imagined_vision
  if args.hold_imagined_vision:
    if args.imagined_vision != "latent":
      parser.error("--hold-imagined-vision requires --imagined-vision latent")
    cfg.run_name += "-held"
  if args.flat_depth_arm:
    if (args.vision_base_checkpoint or args.terrain_height_reset or args.root_waypoints
        or args.imagined_vision != "none"):
      parser.error("--flat-depth-arm configures flat geometry and vision from scratch")
    cfg.flat_depth_arm = args.flat_depth_arm
    cfg.terrain_height = cfg.terrain_lateral_height = 0.0
    cfg.terrain_landscape_seed = None
    cfg.run_name = f"g1-flat-{args.flat_depth_arm}-{args.iterations}it-seed{args.seed}"
    args.imagined_vision = "none" if args.flat_depth_arm == "base" else "latent"
  if args.policy_batch is not None and args.policy_batch != args.batch:
    cfg.policy.batch_size = args.policy_batch
    cfg.run_name += f"-actorbatch{args.policy_batch}"
  if args.actor_microbatch is not None:
    if not 1 <= args.actor_microbatch <= cfg.policy.batch_size:
      parser.error("--actor-microbatch must fit the effective policy batch")
    cfg.actor_microbatch_size = args.actor_microbatch
    cfg.run_name += f"-micro{args.actor_microbatch}"
  if args.training_reference:
    ref = json.loads(args.training_reference.read_text())["runner_cfg"]
    cfg.replay_ratio = ref["replay_ratio"]
    cfg.buffer_size = ref["buffer_size"]
    cfg.wm.lr = ref["wm"]["lr"]
    cfg.wm.num_batches = ref["wm"]["num_batches"]
    cfg.policy.lr = ref["policy"]["lr"]
    cfg.policy.progress_lr_start = ref["policy"]["progress_lr_start"]
    cfg.policy.progress_lr_end = ref["policy"]["progress_lr_end"]
    cfg.policy.progress_lr_steps = ref["policy"]["progress_lr_steps"]
    cfg.log_interval = ref["log_interval"]
    cfg.run_name += "-reference-training"
  clips = None
  if args.lafan40_protocol:
    clips = protocol_clips(args.lafan40_protocol)
    cfg.run_name = cfg.run_name.replace("rough7", "rough-lafan40")
    cfg.policy.horizon = 32
  if args.imagined_vision != "none":
    cfg.imagined_vision = args.imagined_vision
    cfg.run_name += f"-vision-{cfg.imagined_vision}"
    if cfg.critic_variant == "sim-calibrated" and cfg.sim_calibration_weight == 0:
      cfg.run_name = cfg.run_name.replace("-cal0", "-cal0-1rollout")
  if args.vision_base_checkpoint:
    if args.imagined_vision == "none":
      parser.error("--vision-base-checkpoint requires --imagined-vision")
    cfg.init_model_dir = str(args.vision_base_checkpoint.resolve())
    if clips is not None:
      if [Path(p).name for p in source["motion_files"]] != [p.name for p in clips]:
        raise ValueError("Base checkpoint motions differ from the requested LAFAN40 protocol")
    cfg.run_name = cfg.run_name.replace("-scratch-", "-from-base-")
  runner = build_scratch_runner(cfg, args.num_envs, args.seed, clips=clips)
  actor = runner.policy_learner.actor
  initial = {
    name: initial_parameter_hash(module)
    for name, module in (
      ("actor", actor), ("depth", actor.foot_depth_fusion),
      ("dynamics", runner._dynamics), ("critic", runner.policy_learner.critic),
    )
  }
  print(json.dumps({"initial_parameter_sha256": initial}), flush=True)
  if args.flat_depth_arm or args.terrain_curriculum:
    from tensordict import TensorDict

    # Fixed nonzero inputs exercise both leg modulators and the optional visual projection.
    def probe_input(*shape):
      return torch.linspace(-1, 1, math.prod(shape), device=runner.device).reshape(shape)

    width = actor.obs_normalizer.mean.numel()
    if cfg.actor_obs == "proprio_pelvis":
      history = torch.cat((probe_input(4, cfg.history_length, width - 4),
                           probe_input(4, cfg.history_length, 4)), -1)
    else:
      history = probe_input(4, cfg.history_length, width)
    obs = TensorDict({
      "history": history,
      "goal_motion": probe_input(4, actor.task_specs[0].goal_dim),
      "terrain": probe_input(4, 15), "future_feet": probe_input(4, 2, 3, 9),
      "depth_tokens": probe_input(4, 49, 4), "camera_latent": probe_input(4, 32),
    }, batch_size=[4])
    with torch.no_grad():
      action = actor(obs).contiguous().cpu().numpy()
    shared = dict(actor=initial_parameter_hash(actor, exclude_prefix=("visual_action.", "visual_hidden.")),
                  dynamics=initial_parameter_hash(runner._dynamics, exclude_prefix=("vision.", "visual_input.", "visual_delta.")),
                  critic=initial["critic"], action=hashlib.sha256(action.tobytes()).hexdigest())
    print(json.dumps({"shared_initialization": shared}), flush=True)
  if args.expect_control_hashes:
    if not args.root_waypoints:
      parser.error("--expect-control-hashes requires --root-waypoints")
    common = {**initial, "actor": initial_parameter_hash(actor, omit_waypoint_columns=True)}
    if common != json.loads(args.expect_control_hashes.read_text()):
      raise RuntimeError("Shared scratch parameters differ from the no-waypoint control")
    print(json.dumps({"control_parameter_sha256": common}), flush=True)
  if args.expect_initial_hashes and initial != json.loads(args.expect_initial_hashes.read_text()):
    raise RuntimeError("Fresh parameters differ from the seeded smoke initialization")
  runner.run()


if __name__ == "__main__":
  main()
