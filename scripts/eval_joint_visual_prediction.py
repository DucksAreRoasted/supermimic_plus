"""Compare two visual world models open-loop on identical real simulator windows."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from eval_walk_terrain import checkpoint_config
from train_g1_dual_head_depth import build_scratch_runner, protocol_clips

from supermimic.tasks.imagined_vision import unpack_depth


def runner_for(checkpoint, protocol):
  cfg = checkpoint_config(checkpoint)
  cfg.init_model_dir = str(checkpoint.resolve())
  cfg.terrain_curriculum = ()
  cfg.terrain_landscape_seed = 1702
  cfg.buffer_size = 96
  cfg.policy.batch_size = 64
  cfg.policy.noise = 0.
  return build_scratch_runner(cfg, 64, 42, clips=protocol_clips(protocol))


def collect(runner):
  runner.env.reset()
  runner._seed_sim_history()
  # Preserve checkpoint normalization statistics throughout this evaluation.
  with (
    patch.object(runner, "_flush_normalization", runner._pending_norm_updates.clear),
    patch.object(runner.logger, "process_env_step", lambda *_: None),
  ):
    for _ in range(96):
      runner.step()
  return runner.replay_buffer.sample(33, 128).clone()


@torch.no_grad()
def score(runner, batch):
  rep, wm = runner._representation, runner._dynamics
  truth = rep.represent(batch["state"])
  aux_truth = rep.extract_aux_state(batch["state"])
  encoded = wm.vision.encode(unpack_depth(batch["camera_pixels"]))
  state, aux, z = truth[:, 0], aux_truth[:, 0], encoded[:, 0]
  runner.world_model_learner.depth_patch.bind_replay(batch[:, 0])
  metrics = {}
  for t in range(1, 33):
    action, ctx = batch["action"][:, t - 1], batch["sim_ctx"][:, 0]
    if wm.visual_input is None:
      aux, state = wm(state, action, aux, ctx)
      z = wm.vision.next_latent(z, state, action)
    else:
      aux, state, z = wm.joint_step(state, action, aux, ctx, z)
    if t in (1, 8, 16, 32):
      error = (z - encoded[:, t]).square().mean()
      persistence = (encoded[:, 0] - encoded[:, t]).square().mean()
      metrics[t] = dict(root_xy_m=(aux[:, :2] - aux_truth[:, t, :2]).norm(dim=-1).mean().item(),
                        state_rmse=(state - truth[:, t]).square().mean().sqrt().item(),
                        latent_mse=error.item(), latent_vs_copy=(error / persistence.clamp_min(1e-9)).item())
  return metrics


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for arg in ("control", "joint", "protocol", "output"):
    p.add_argument(arg, type=Path)
  a = p.parse_args()
  runs = []
  try:
    for path in (a.control, a.joint):
      runs.append(runner_for(path, a.protocol))
    batches = [collect(r) for r in runs]
    batch = torch.cat(batches, 0)
    assert not (batch["terminated"][:, :-1] | batch["truncated"][:, :-1]).any()
    result = {name: score(r, batch) for name, r in zip(("control", "joint"), runs, strict=True)}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
  finally:
    for r in runs:
      r.env.close()


if __name__ == "__main__":
  main()
