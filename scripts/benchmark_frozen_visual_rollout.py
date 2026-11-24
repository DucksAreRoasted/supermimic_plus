"""Time the 32-step visual policy objective and its first-action derivative with fixed weights."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from eval_walk_terrain import checkpoint_config
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict
from train_g1_dual_head_depth import build_scratch_runner, protocol_clips

from supermimic.tasks.imagined_vision import pack_depth
from supermimic.utils.warp_fk import WarpForwardKinematics


@torch.no_grad()
def fixed_starts(runner, batch_size):
  runner.env.reset()
  runner._seed_sim_history()
  obs = runner.env.get_observations()
  state = obs["full_state"]
  times = runner.motion_cmd.time_steps.clone()
  ctx = runner._sim_context.read(runner.env.unwrapped)
  rep = runner._representation
  runner._build_sim_obs_td(
    rep.represent(state), rep.extract_aux_state(state), obs["actor"], times, ctx
  )
  aligned = runner.cfg.camera_profile == "instinct"
  rows = TensorDict(
    {
      "state": state,
      "motion_time": times,
      "prev_action": runner.action_manager.action,
      "sim_ctx": ctx,
      "depth_image": runner.sim_depth.image.flatten(1).half(),
      "depth_origin": runner.sim_depth.origin,
      "camera_pixels": runner._camera_pixels if aligned else pack_depth(
        runner.env.unwrapped.scene["walk_depth"].data.distances
      ),
    },
    batch_size=[len(state)],
  )
  if aligned:
    rows["camera_env"] = torch.arange(len(state), device=state.device)
  if runner.cfg.history_length > 1:
    length = runner.cfg.history_length
    rows["sensor_window"] = obs["actor"][:, None].expand(-1, length, -1)
    rows["pad_mask"] = (torch.arange(length, device=state.device) < length - 1).expand(len(state), -1)
  return rows[torch.arange(batch_size, device=state.device) % len(state)].clone()


def measure(runner, starts, repeats):
  policy = runner.policy_learner
  times = []
  for _ in range(1 + repeats):
    torch.manual_seed(42)
    perturbation = torch.zeros_like(starts["prev_action"], requires_grad=True)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    obs = runner.wm_env.reset(starts)
    discounted, survival = 0.0, 1.0
    for step in range(policy.cfg.horizon):
      action = policy.actor(obs)
      if step == 0:
        action = action + perturbation
      obs, reward, done, _ = runner.wm_env.step(action)
      discounted = discounted + policy.cfg.gamma**step * survival * reward
      survival = survival * (1 - done)
    objective = (discounted + policy._bootstrap(obs, survival, done)).mean()
    (gradient,) = torch.autograd.grad(objective, perturbation)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - begin)
  return statistics.median(times[1:]), objective.detach(), gradient


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--batch", type=int, default=1024)
  parser.add_argument("--repeats", type=int, default=4)
  parser.add_argument(
    "--protocol",
    type=Path,
    default=Path(__file__).resolve().parents[1]
    / "docs/experiments/lafan40_seed42.json",
  )
  args = parser.parse_args()
  if min(args.batch, args.repeats) < 1:
    parser.error("batch and repeats must be positive")
  if (
    os.getenv("SUPERMIMIC_DISABLE_COMPILE") != "1"
    or os.getenv("TORCH_COMPILE_DISABLE") != "1"
  ):
    parser.error(
      "Set SUPERMIMIC_DISABLE_COMPILE=1 and TORCH_COMPILE_DISABLE=1 for the eager visual comparison"
    )
  cfg = checkpoint_config(args.checkpoint)
  if (
    cfg.imagined_vision not in ("render", "latent")
    or not cfg.policy.survival_masking
  ):
    parser.error("Expected a visual checkpoint with survival masking")
  cfg.init_model_dir = str(args.checkpoint.resolve())
  cfg.policy.batch_size = args.batch
  cfg.buffer_size = cfg.wm.horizon
  clips = protocol_clips(args.protocol)
  manifest = json.loads((args.checkpoint / "run_manifest.json").read_text())
  if [p.name for p in clips] != [Path(p).name for p in manifest["motion_files"]]:
    parser.error("Checkpoint and protocol motion order differ")
  configure_torch_backends()
  runner = build_scratch_runner(cfg, min(args.batch, 16), 42, clips=clips)
  try:
    for module in (
      runner.policy_learner.actor,
      runner.policy_learner.critic,
      runner._dynamics,
      runner._representation,
    ):
      module.eval().requires_grad_(False)
    starts = fixed_starts(runner, args.batch)
    fused = runner._representation.fk
    if not isinstance(fused, WarpForwardKinematics):
      raise RuntimeError("The visual runner did not select fused kinematics")
    runner._representation.fk = fused.fk
    old_time, old_return, old_gradient = measure(runner, starts, args.repeats)
    runner._representation.fk = fused
    new_time, new_return, new_gradient = measure(runner, starts, args.repeats)
    torch.testing.assert_close(new_return, old_return, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_gradient, old_gradient, rtol=3e-4, atol=1e-5)
    print(
      json.dumps(
        dict(
          mode=cfg.imagined_vision,
          batch=args.batch,
          horizon=cfg.policy.horizon,
          device=torch.cuda.get_device_name(),
          mjlab_seconds=old_time,
          fused_seconds=new_time,
          return_difference=(new_return - old_return).abs().item(),
          max_gradient_difference=(new_gradient - old_gradient).abs().max().item(),
        ),
        allow_nan=False,
      )
    )
  finally:
    runner.env.close()


if __name__ == "__main__":
  main()
