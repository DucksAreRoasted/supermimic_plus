"""Compare the Shared residual linear and nonlinear heads on the same flat LAFAN40 task."""

import argparse
import hashlib
import json
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device


def parameter_digest(module):
  digest = hashlib.sha256()
  for name, param in module.named_parameters():
    if not name.startswith("head_mlps."):
      digest.update(name.encode() + b"\0")
      digest.update(param.detach().contiguous().cpu().numpy().tobytes())
  return digest.hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("head", choices=("linear", "nonlinear"))
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--iterations", type=int, default=20_000)
  parser.add_argument("--initial-parameters", type=Path, help="Record or verify common initial weights across arms")
  parser.add_argument("--smoke", action="store_true", help="16 iterations and reduced batch sizes")
  args = parser.parse_args()
  if args.head == "linear" and args.initial_parameters is not None:
    if (args.initial_parameters.parent / "linear.cancelled").is_file():
      parser.exit(75, "Linear arm cancelled for this run; skipping before environment creation.\n")
  protocol = json.loads(args.protocol.read_text())
  clips = protocol["clips"]
  if len({c["path"] for c in clips}) != 40 or len(clips) != 40 or protocol["seed"] != 42:
    raise ValueError("Expected the seed-42 LAFAN40 protocol")
  for clip in clips:
    with Path(clip["path"]).open("rb") as source:
      if hashlib.file_digest(source, "sha256").hexdigest() != clip["sha256"]:
        raise ValueError(f"Motion differs from the flat protocol: {clip['path']}")
  iterations = 16 if args.smoke else args.iterations
  if iterations < 1:
    parser.error("--iterations must be positive")
  envs = 64 if args.smoke else 4096
  task = "SuperMimic-GenCoord-G1"
  device = setup_device("cuda:0")
  env_cfg = load_env_cfg(task, play=False)
  env_cfg.seed = 42
  torch.manual_seed(42)
  env_cfg.commands["motion"].motion_file = [c["path"] for c in clips]
  env_cfg.commands["motion"].sampling_mode = "external"
  label = "smoke" if args.smoke else "scratch"
  cfg = SuperMimicRunnerCfg(
    run_name=f"g1-flat-lafan40-shared-{args.head}-noprior-cal0-{label}-{iterations}it-seed42",
    critic_variant="sim-calibrated",
    sim_calibration_weight=0.0,
    sim_calibration_horizon=32,
    sim_calibration_interval=1,
    sim_calibration_epochs=2,
    residual_correction=True,
    residual_shared_trunk=True,
    shared_head_width=128 if args.head == "nonlinear" else 0,
    latent_prior_weight=0.0,
    latent_prior_blend=0.0,
    replay_ratio=16,
    log_interval=1 if args.smoke else 10,
    save_interval=1000,
    keep_iteration_checkpoints=not args.smoke,
    max_iterations=iterations,
  )
  cfg.wm.horizon, cfg.policy.horizon = 8, 32
  cfg.wm.num_batches, cfg.policy.num_batches = (2 if args.smoke else 10), 1
  cfg.wm.batch_size = 32 if args.smoke else 2048
  cfg.policy.batch_size = envs
  cfg.return_sampler.focus_share = 0.4
  cfg.configure_terrain(env_cfg)
  env = create_env(env_cfg, envs, device)
  try:
    runner = SuperMimicRunner(env, cfg, task, device=device)
    common = {
      "actor": parameter_digest(runner.policy_learner.actor),
      "critic": parameter_digest(runner.policy_learner.critic),
      "common_wm": parameter_digest(runner._dynamics),
    }
    print("Initial common parameters " + json.dumps(common), flush=True)
    if args.initial_parameters is not None:
      try:
        with args.initial_parameters.open("x") as output:
          json.dump(common, output, indent=2)
      except FileExistsError:
        if json.loads(args.initial_parameters.read_text()) != common:
          raise ValueError("The two runs have different common initial parameters") from None
    runner.run()
    if args.smoke:
      if common["actor"] == parameter_digest(runner.policy_learner.actor):
        raise RuntimeError("The physical policy did not update")
      for head in runner._dynamics.head_mlps:
        norm = head[-1].weight.detach().norm().item()
        if not 0.0 < norm < float("inf"):
          raise RuntimeError("A zero-initialized nonlinear output head did not learn")
      print("Verified smoke updates", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
