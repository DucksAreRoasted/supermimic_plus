"""Loop a recorded physical trajectory and its reference in a native MuJoCo window."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("recording", type=Path, help="Directory containing scene.mjb and trajectory.npz")
  args = parser.parse_args()
  model = mujoco.MjModel.from_binary_path(str(args.recording / "scene.mjb"))
  with np.load(args.recording / "trajectory.npz") as data:
    actual, reference, dt = data["actual"], data["reference"], float(data["dt"])
  if (actual.shape != reference.shape or actual.ndim != 2 or len(actual) < 2
      or actual.shape[1] != model.nq or not np.isfinite(dt) or dt <= 0
      or not (np.isfinite(actual).all() and np.isfinite(reference).all())):
    raise ValueError("Invalid recording or incompatible MuJoCo model")
  robot, ghost = mujoco.MjData(model), mujoco.MjData(model)
  options, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
  robot.qpos[:] = actual[0]
  mujoco.mj_forward(model, robot)
  with mujoco.viewer.launch_passive(model, robot, show_left_ui=False, show_right_ui=False) as viewer:
    travel = reference[-1, :2] - reference[0, :2]
    middle = (actual[:, :3] + reference[:, :3]) / 2
    with viewer.lock():
      viewer.cam.azimuth = np.degrees(np.arctan2(travel[1], travel[0])) + 74
      viewer.cam.elevation = -14
      viewer.cam.distance = 3.25
      viewer.cam.lookat[:] = middle[0]
    start = time.monotonic()
    previous = 0
    ticks = 0
    while viewer.is_running():
      index = int((time.monotonic() - start) / dt) % len(actual)
      with viewer.lock():
        robot.qpos[:] = actual[index]
        ghost.qpos[:] = reference[index]
        mujoco.mj_forward(model, robot)
        mujoco.mj_forward(model, ghost)
        viewer.cam.lookat[:] += middle[index] - middle[previous]
        scene = viewer.user_scn
        scene.ngeom = 0
        mujoco.mjv_addGeoms(model, ghost, options, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for i in range(scene.ngeom):
          geom = scene.geoms[i]
          geom.matid = -1
          geom.rgba[:] = (0.10, 0.95, 0.32, 0.40)
        ghost_geoms = scene.ngeom
      viewer.sync()
      if ticks == 24:
        print(f"Playing {len(actual)} frames at {1/dt:.1f} fps with {ghost_geoms} green reference geoms", flush=True)
      ticks += 1
      previous = index
      time.sleep(min(dt / 2, 1 / 60))


if __name__ == "__main__":
  main()
