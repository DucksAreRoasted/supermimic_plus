"""Capture the independent real-G1 device step adapter; no training."""

from __future__ import annotations

import glob

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.mjlab_cuda_graph_env import DeviceStepAdapter


def main() -> None:
  assert torch.cuda.is_available()
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.scene.num_envs = 16
  cfg.commands["motion"].motion_file = glob.glob(
    "data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk1_subject1.npz"
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
  env.reset()
  adapter = DeviceStepAdapter(env)
  action = torch.zeros_like(adapter.action)
  for _ in range(2):
    adapter.step_core_device(action)
  env.reset()
  stream = torch.cuda.Stream()
  stream.wait_stream(torch.cuda.current_stream())
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph, stream=stream):
    out = adapter.step_core_device(action)
    adapter.reset_masked(adapter.reset_mask)
  graph.replay()
  torch.cuda.synchronize()
  assert all(torch.isfinite(x).all() for x in out if x.dtype.is_floating_point)
  print("PASS device step + masked reset CUDA Graph capture")
  env.close()


if __name__ == "__main__":
  main()
