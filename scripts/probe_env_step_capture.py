"""Probe capture of the real mjlab env.step; never used by training."""
from __future__ import annotations

import glob
import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import supermimic.tasks  # noqa: F401


def main() -> None:
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.scene.num_envs = 16
  cfg.commands["motion"].motion_file = glob.glob(
    "data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk1_subject1.npz"
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
  env.reset()
  action = torch.zeros(
    (env.num_envs, env.action_manager.total_action_dim), device="cuda"
  )
  for _ in range(2):
    env.step(action)
  env.reset()
  stream = torch.cuda.Stream()
  stream.wait_stream(torch.cuda.current_stream())
  graph = torch.cuda.CUDAGraph()
  try:
    with torch.cuda.graph(graph, stream=stream):
      env.step(action)
    graph.replay()
    torch.cuda.synchronize()
  except Exception as exc:
    print(f"EXPECTED_CAPTURE_BLOCKER: {type(exc).__name__}: {exc}")
    raise SystemExit(2)
  print("PASS real env.step capture")


if __name__ == "__main__":
  main()
