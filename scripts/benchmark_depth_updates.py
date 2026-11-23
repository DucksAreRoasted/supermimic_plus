"""Benchmark depth-policy learner updates or the complete 4096-environment training loop."""

import argparse
import json
import os
import time
from pathlib import Path
from statistics import median

import torch
from mjlab.utils.torch import configure_torch_backends
from train_g1_dual_head_depth import (
  build_scratch_runner,
  initial_parameter_hash,
  scratch_config,
)


def check_reference_cache(runner):
  feet = runner.future_feet
  ends = feet.motion.motion_ends
  sample = torch.linspace(
    0, feet.motion.joint_pos.shape[0] - 1, 257, device=ends.device
  ).long()
  times = torch.cat([sample, ends - 1, ends[:-1]])
  cached = (*runner.wm_env.reference_body_state(times), feet(times))
  direct = (*feet.motion.body_state(times), feet.compute_feet(times))
  errors = {}
  for name, value, expected in zip(
    ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel", "future_feet"),
    cached,
    direct,
    strict=True,
  ):
    torch.testing.assert_close(value, expected, rtol=1e-5, atol=2e-6)
    errors[name] = (value - expected).abs().max().item()
  return errors


def benchmark_full_runner(args):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  if args.no_update_graphs:
    raise ValueError("Full-run timings use the production learner graphs")
  cfg = scratch_config(
    args.iterations,
    args.batch,
    42,
    False,
    4096,
    official_backends=args.official_backends,
    cache_reference=args.cache_reference,
  )
  cfg.run_name += f"-speed-smoke-replay{args.replay_capacity}"
  cfg.buffer_size = args.replay_capacity
  cfg.log_interval = 4
  runner = build_scratch_runner(cfg, 4096, 42)
  try:
    initial_hash = initial_parameter_hash(runner.policy_learner.actor)
    cache_errors = check_reference_cache(runner) if args.cache_reference else None
    runner.run()
    writer = runner.logger.writer
    writer.flush()
    events = EventAccumulator(writer.log_dir, size_guidance={"scalars": 0}).Reload()
    metrics = {}
    for tag in (
      "Perf/collection_time",
      "Perf/wm_learning_time",
      "Perf/policy_learning_time",
    ):
      metrics[tag] = {
        e.step: e.value / cfg.log_interval for e in events.Scalars(tag) if e.step >= 20
      }
    windows = sorted(metrics["Perf/collection_time"])
    seconds = [sum(series[step] for series in metrics.values()) for step in windows]
    stamps = {e.step: e.wall_time for e in events.Scalars("Perf/total_fps")}
    wall_seconds = [
      (stamps[step] - stamps[step - cfg.log_interval]) / cfg.log_interval
      for step in windows
    ]
    grads = {
      tag: events.Scalars(tag)[-1].value
      for tag in (
        "Policy/grad_norm",
        "Policy/critic_grad_norm",
        "WM/grad_main_output",
        "WM/grad_residual_output",
      )
    }
    if not all(0 < g < float("inf") for g in grads.values()):
      raise RuntimeError(f"Unexpected gradients {grads}")
    if initial_hash == initial_parameter_hash(runner.policy_learner.actor):
      raise RuntimeError("Actor weights did not change")
    wm_graph = runner.world_model_learner._captured_update._graph is not None
    policy_graph = runner.policy_learner._captured_update._graph is not None
    if not (wm_graph and policy_graph):
      raise RuntimeError("The full-run smoke ended before both learner graphs were captured")
    return dict(
      batch=args.batch,
      num_envs=4096,
      replay_capacity=cfg.buffer_size,
      iterations=cfg.max_iterations,
      recent_intervals_ending_at=windows,
      seconds_per_iteration=seconds,
      median_seconds=median(seconds),
      transitions_per_second=4096 / median(seconds),
      wall_seconds_per_iteration=wall_seconds,
      wall_transitions_per_second=4096 / median(wall_seconds),
      median_seconds_by_stage={tag: median(v.values()) for tag, v in metrics.items()},
      official_backends=args.official_backends,
      compile_disabled=os.getenv("TORCH_COMPILE_DISABLE"),
      initial_actor_sha256=initial_hash,
      last_gradients=grads,
      ref_cache_max_abs=cache_errors,
      wm_graph=wm_graph,
      policy_graph=policy_graph,
      peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
      log_dir=writer.log_dir,
    )
  finally:
    if runner.logger.writer:
      runner.logger.writer.close()
    runner.env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("out", type=Path)
  parser.add_argument("--batch", type=int, required=True)
  parser.add_argument("--updates", type=int, default=8)
  parser.add_argument("--cache-reference", action="store_true")
  parser.add_argument(
    "--full-runner",
    action="store_true",
    help="Time the production loop with 4096 real camera-equipped environments",
  )
  parser.add_argument("--iterations", type=int, default=41)
  parser.add_argument(
    "--replay-capacity",
    type=int,
    default=64,
    help="Replay rounds allocated for the full-run smoke",
  )
  parser.add_argument("--official-backends", action="store_true")
  parser.add_argument(
    "--no-update-graphs",
    action="store_true",
    help="Measure eager learners as a diagnostic control",
  )
  args = parser.parse_args()
  if min(args.batch, args.updates) < 1:
    parser.error("batch and updates must be positive")
  if args.full_runner and (
    args.iterations < 25 or args.replay_capacity < min(args.iterations, 512)
  ):
    parser.error(
      "Use >=25 iterations and enough capacity to retain the same history as the 512-round replay"
    )
  if args.official_backends:
    configure_torch_backends()
  free, total = torch.cuda.mem_get_info()
  # The production trainer shares this card. Leave room for non-PyTorch allocations.
  budget = free - 4 * 1024**3
  if budget <= 0:
    raise RuntimeError("Insufficient free GPU memory for the benchmark")
  torch.cuda.set_per_process_memory_fraction(budget / total)
  if args.full_runner:
    result = benchmark_full_runner(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
    return
  num_envs = 64
  cfg = scratch_config(
    40000,
    args.batch,
    42,
    False,
    num_envs,
    official_backends=args.official_backends,
    cache_reference=args.cache_reference,
  )
  runner = build_scratch_runner(cfg, num_envs, 42)
  try:
    runner.logger.set_log_dir(str(args.out.parent / (args.out.stem + "-events")))
    wm, policy = runner.world_model_learner, runner.policy_learner
    if args.no_update_graphs:
      wm._captured_update = wm._update_from
      policy._captured_update = policy._update_from
    actor_hash = initial_parameter_hash(policy.actor)
    cache_errors = check_reference_cache(runner) if args.cache_reference else None
    runner.env.reset()
    runner._seed_sim_history()
    for _ in range(32):
      runner.step()
    assert runner.replay_buffer.num_transitions >= cfg.wm.horizon
    policy_seconds, wm_seconds = [], []
    for i in range(5 + args.updates):
      torch.cuda.synchronize()
      begin = time.perf_counter()
      wm_logs = wm.update(
        runner.replay_buffer.sample(cfg.wm.horizon, cfg.wm.batch_size)
        for _ in range(cfg.wm.num_batches)
      )
      torch.cuda.synchronize()
      middle = time.perf_counter()
      policy_logs = policy.update()
      runner._update_ema_actor()
      torch.cuda.synchronize()
      end = time.perf_counter()
      values = torch.stack(
        [v.reshape(()) for v in (*wm_logs.values(), *policy_logs.values())]
      )
      if not torch.isfinite(values).all():
        raise FloatingPointError(f"Non-finite statistic in update {i}")
      if i >= 5:
        policy_seconds.append(end - middle)
        wm_seconds.append(middle - begin)
    rate = 4096 * cfg.replay_ratio / (cfg.wm.num_batches * args.batch * cfg.wm.horizon)
    gradients = {
      k: float(v) for k, v in (wm_logs | policy_logs).items() if "grad" in k.lower()
    }
    assert (
      gradients["Policy/grad_norm"] > 0 and gradients["Policy/critic_grad_norm"] > 0
    )
    assert (
      gradients["WM/grad_main_output"] > 0 and gradients["WM/grad_residual_output"] > 0
    )
    assert actor_hash != initial_parameter_hash(policy.actor)
    result = dict(
      batch=args.batch,
      updates_timed=args.updates,
      updates_per_4096=rate,
      actor_starts_per_4096=rate * cfg.policy.batch_size,
      wm_sequences_per_4096=rate * cfg.wm.num_batches * cfg.wm.batch_size,
      policy_horizon=cfg.policy.horizon,
      wm_horizon=cfg.wm.horizon,
      initial_actor_sha256=actor_hash,
      ref_cache_max_abs=cache_errors,
      official_backends=args.official_backends,
      matmul_precision=torch.backends.cuda.matmul.fp32_precision,
      compile_disabled=os.getenv("TORCH_COMPILE_DISABLE"),
      wm_graph=getattr(wm._captured_update, "_graph", None) is not None,
      policy_graph=getattr(policy._captured_update, "_graph", None) is not None,
      wm_seconds=wm_seconds,
      policy_seconds=policy_seconds,
      gradients=gradients,
      normalized_learn_seconds=rate
      * median([w + p for w, p in zip(wm_seconds, policy_seconds, strict=True)]),
      peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
      json.dumps(
        {
          k: v
          for k, v in result.items()
          if k not in ("wm_seconds", "policy_seconds", "gradients")
        }
      )
    )
  finally:
    if runner.logger.writer:
      runner.logger.writer.close()
    runner.env.close()


if __name__ == "__main__":
  main()
