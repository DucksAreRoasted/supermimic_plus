"""Play a SuperMimic multi-encoder actor against the sim."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)


@dataclass
class PlayConfig:
  wandb_path: str | None = None
  model_dir: Path | None = None
  """Local checkpoint directory, using the standard G1 runner architecture."""
  motion_file: str | None = None
  initial_task: Literal["motion", "velocity"] = "motion"
  """Task selected when the viewer opens. Switch live via the GUI dropdown."""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"
  world_model_ghost: bool = False
  """Also show world-model predictions; the green ghost is always the reference."""
  max_steps: int | None = None
  """Close after this many simulation steps; None keeps the viewer running."""


def play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  # A single world cannot borrow spare contact capacity from other worlds.
  env_cfg.sim.nconmax = max(env_cfg.sim.nconmax or 0, 256)
  env_cfg.sim.njmax = max(env_cfg.sim.njmax or 0, 1024)
  env_cfg.observations["actor"].enable_corruption = True

  if (cfg.model_dir is None) == (cfg.wandb_path is None):
    raise ValueError("Provide exactly one of --model-dir or --wandb-path")
  if cfg.model_dir is not None:
    model_dir = cfg.model_dir.expanduser().resolve()
    for filename in ("actor.pt", "world_model.pt"):
      if not (model_dir / filename).is_file():
        raise FileNotFoundError(model_dir / filename)
    if cfg.motion_file is None:
      raise ValueError("Local playback requires --motion-file")
    runner_cfg = SuperMimicRunnerCfg()
    if "MaxCoord" in task_id:
      runner_cfg.representation = "maxcoord"
    motion_artifacts = []
    print(f"[INFO] Local checkpoint: {model_dir}")
    print("[INFO] Using standard runner architecture; loading weights strictly")
  else:
    import wandb

    print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
    run = wandb.Api().run(cfg.wandb_path)
    runner_cfg = load_runner_cfg(run.config)
    motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
    model_dir = download_model_dir(run)
  check_representation_matches(task_id, runner_cfg.representation)

  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  command_term.debug_vis = True
  command_term.viz.mode = "ghost"
  command_term.viz.ghost_color = (0.2, 0.9, 0.4, 0.35)
  motion_names: list[str] = []
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
    command_term.motion_file = files
    motion_names = [Path(f).stem for f in files]
    print(f"[INFO] Using {len(files)} local motion(s)")
  elif motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(download_artifact_cached(artifact) / "motion.npz"))
    command_term.motion_file = (
      motion_files if len(motion_files) > 1 else motion_files[0]
    )
    motion_names = [a.name.split(":")[0] for a in motion_artifacts]
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  wm = build_world_model(runner_cfg, env, device)
  representation = wm.representation
  dynamics = wm.dynamics

  wm.load_bundle(
    torch.load(model_dir / "world_model.pt", map_location=device, weights_only=True)
  )
  dynamics.eval()

  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(
    torch.load(model_dir / "actor.pt", map_location=device, weights_only=True)
  )
  actor.eval()

  wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer
  print(f"[INFO] Starting viewer: {resolved}")
  print("[INFO] Solid robot: policy; green translucent robot: reference motion")

  if resolved == "native":
    SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
      show_wm_prediction=cfg.world_model_ghost,
    ).run(num_steps=cfg.max_steps)
  else:
    SuperMimicViserViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
      show_wm_prediction=cfg.world_model_ghost,
    ).run(num_steps=cfg.max_steps)

  env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(PlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
  play(chosen_task, args)


if __name__ == "__main__":
  main()
