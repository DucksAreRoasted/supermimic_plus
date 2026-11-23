"""Reload a visual checkpoint and verify actual sensor, imagined observations and Actor gradients."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from eval_walk_terrain import checkpoint_config
from train_g1_dual_head_depth import build_scratch_runner


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  ckpt = args.checkpoint
  cfg = checkpoint_config(ckpt)
  # Reload the trained weights below; probing should not need the parent checkpoint.
  cfg.init_model_dir = None
  motion_files = json.loads((ckpt / "run_manifest.json").read_text()).get("motion_files")
  clips = [Path(p) for p in motion_files] if isinstance(motion_files, list) else None
  cfg.policy.batch_size = cfg.wm.batch_size = 16
  runner = build_scratch_runner(cfg, 16, 42, clips=clips)
  temp = tempfile.TemporaryDirectory()
  runner.logger.set_log_dir(temp.name)
  try:
    actor = runner.policy_learner.actor
    actor.load_state_dict(
      torch.load(ckpt / "actor.pt", map_location="cuda", weights_only=True)
    )
    runner._wm.load_bundle(
      torch.load(ckpt / "world_model.pt", map_location="cuda", weights_only=True)
    )
    actor.eval()
    env = runner.wm_env
    vision = env.dynamics.vision
    vision.eval()
    mode = cfg.imagined_vision
    assert mode in ("latent", "render") and (
      (env.imagined_camera is not None) == (mode == "render")
    )
    assert (
      cfg.shared_head_width == 128
      and cfg.residual_shared_trunk
      and cfg.sim_calibration_weight == cfg.latent_prior_weight == 0
    )
    runner.env.reset()
    runner._seed_sim_history()
    for _ in range(3):
      runner.step()
    # Compare a noiseless render against the exact camera pose underlying the real sensor buffer.
    report = dict(mode=mode, horizon=cfg.policy.horizon)
    camera_calls = []
    if mode == "render":
      data = runner.env.unwrapped.scene["walk_depth"].data
      root = torch.cat([data.pos_w, data.quat_w], -1)
      render = env.imagined_camera.ranges
      real = data.distances.clone()
      rendered = render(root)
      shared = (real > 0) & (rendered > 0)
      report.update(
        hit_mask_agreement=float(((real > 0) == (rendered > 0)).float().mean()),
        compared_hits=int(shared.sum()),
        range_mae_mm=float((real[shared] - rendered[shared]).abs().mean() * 1000),
      )
      assert (
        shared.any()
        and report["hit_mask_agreement"] > 0.995
        and report["range_mae_mm"] < 1
      )

      def counted(pose):
        camera_calls.append(pose.detach().clone())
        return render(pose)

      env.imagined_camera.ranges = counted
    with torch.no_grad():
      batch = env.sample_initial_states()
      obs = env.reset(batch)
      starting_z = obs["camera_latent"].clone()
      delta_action = []
      forecast_err = []
      for _ in range(cfg.policy.horizon):
        z = obs["camera_latent"].clone()
        action = actor(obs)
        masked = obs.clone(False)
        masked["camera_latent"] = torch.zeros_like(z)
        delta_action.append((action - actor(masked)).abs().mean())
        obs, reward, term, _ = env.step(action)
        if mode == "latent":
          forecast_err.append(
            (vision.next_latent(z, env.rep_state, action) - obs["camera_latent"])
            .abs()
            .max()
          )
        assert (
          not obs["camera_latent"].requires_grad
          and torch.isfinite(reward).all()
          and torch.isfinite(term).all()
        )
      report.update(
        imagination_renders=len(camera_calls),
        latent_change=float((obs["camera_latent"] - starting_z).abs().mean()),
        action_change_on_latent_ablation=float(torch.stack(delta_action).mean()),
      )
      assert len(camera_calls) == (cfg.policy.horizon + 1 if mode == "render" else 0)
      assert (
        report["latent_change"] > 0 and report["action_change_on_latent_ablation"] > 0
      )
      report["terminal_latent_rms"] = float(obs["camera_latent"].square().mean().sqrt())
      if forecast_err:
        assert 0.95 < report["terminal_latent_rms"] < 1.05
        report["recurrence_max_difference"] = float(torch.stack(forecast_err).max())
        assert report["recurrence_max_difference"] < 1.0e-6
    # Supply a live feature leaf: Actor BPTT must update the projection but never the encoder feature.
    obs = obs.clone(False)
    z = obs["camera_latent"].detach().requires_grad_()
    obs["camera_latent"] = z
    gz, gw = torch.autograd.grad(
      actor(obs).sum(), (z, actor.visual_action.weight), allow_unused=True
    )
    assert gz is None and gw.norm() > 0
    report["actor_feature_gradient_blocked"] = True
    report["actor_projection_gradient_norm"] = float(gw.norm())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)
  finally:
    runner.logger.writer.close()
    temp.cleanup()
    runner.env.close()


if __name__ == "__main__":
  main()
