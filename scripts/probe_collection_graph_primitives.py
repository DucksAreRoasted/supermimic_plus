"""Independent CUDA-graph probes for the mjlab collection prerequisites.

This deliberately does not call ``env.step`` or start training. It checks the
fixed device buffers and the two capture-safe primitives needed before wrapping
the full collection loop.
"""

from __future__ import annotations

import glob

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401


def _capture(fn, *inputs):
  stream = torch.cuda.current_stream()
  for _ in range(2):
    fn(*inputs)
  torch.cuda.synchronize()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph, stream=stream):
    fn(*inputs)
  graph.replay()
  torch.cuda.synchronize()


def main() -> None:
  assert torch.cuda.is_available(), "CUDA is required"
  device = torch.device("cuda:0")
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.scene.num_envs = 16
  cfg.commands["motion"].motion_file = glob.glob(
    "data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk1_subject1.npz"
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()

  command = env.command_manager.get_term("motion")
  body_indexes = command.body_indexes
  assert body_indexes.is_cuda and body_indexes.dtype == torch.long
  print(f"PASS body index: device={body_indexes.device} dtype={body_indexes.dtype}")

  action_manager = env.action_manager
  action = torch.zeros_like(action_manager.action)
  robot = env.scene["robot"]
  ctrl_ids = robot.indexing.ctrl_ids.to(device=device, dtype=torch.long)
  staged = torch.zeros(
    (env.num_envs, int(env.sim.mj_model.nu)), device=device, dtype=action.dtype
  )
  _capture(lambda x: staged.index_copy_(1, ctrl_ids, x), action)
  assert staged.is_cuda and staged.shape[1] == int(env.sim.mj_model.nu)
  print(f"PASS fixed actuator buffer: shape={tuple(staged.shape)} ids={ctrl_ids.numel()}")

  # Verify the history/reset primitive with a fixed-size mask. No nonzero()
  # or host-side env-id list is allowed inside this capture.
  state = torch.zeros((env.num_envs, 4), device=device)
  reset_state = torch.ones_like(state)
  reset_mask = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=device)
  reset_mask[: env.num_envs // 2] = True

  def masked_reset():
    state.copy_(torch.where(reset_mask, reset_state, state))

  _capture(masked_reset)
  assert torch.all(state[: env.num_envs // 2] == 1)
  print("PASS masked reset: fixed-shape device mask")
  env.close()
  print("PASS collection graph primitive smoke")


if __name__ == "__main__":
  main()
