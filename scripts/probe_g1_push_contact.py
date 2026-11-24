"""Check whether prescribed G1 palms measurably push a free box in native MuJoCo."""

import argparse
import json

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

from supermimic.tasks.push_box import PushBox


def rollout(mass: float, lateral_offset: float):
  spec = get_spec()
  spec.worldbody.add_geom(
    name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(4, 4, 0.1)
  )
  PushBox(center=(0.62, lateral_offset, 0.42), mass=mass)(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  pose = data.qpos.copy()
  robot_dofs = model.joint("push_box_free").dofadr[0]
  box_q = model.joint("push_box_free").qposadr[0]
  # Raise both arms into a pushing posture, using the actual palm sites and G1 axes.
  for side, y in (("left", 0.19), ("right", -0.19)):
    palm = model.site(side + "_palm").id
    joint_ids = [
      i
      for i in range(model.njnt)
      if model.joint(i).name.startswith(side + "_")
      and any(s in model.joint(i).name for s in ("shoulder", "elbow", "wrist"))
    ]
    q, v = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    target = [0.33, y, 0.81]
    jac = np.empty((3, model.nv))
    for _ in range(100):
      mujoco.mj_forward(model, data)
      error = target - data.site_xpos[palm]
      if np.linalg.norm(error) < 0.0002:
        break
      mujoco.mj_jacSite(model, data, jac, None, palm)
      j = jac[:, v]
      step = j.T @ np.linalg.solve(j @ j.T + 0.0005 * np.eye(3), error)
      data.qpos[q] = np.clip(
        data.qpos[q] + np.clip(step, -0.1, 0.1), *model.jnt_range[joint_ids].T
      )
    if np.linalg.norm(error) > 0.005:
      raise ValueError(f"{side} palm unreachable: {error}")
  pose[:box_q] = data.qpos[:box_q]
  hands = {model.geom(s + "_hand_collision").id for s in ("left", "right")}
  box = model.geom("push_box_collision").id
  floor = model.geom("ground").id
  contact_hands = set()
  body_contacts = set()
  start = data.qpos[box_q : box_q + 3].copy()
  for _ in range(round(0.85 / model.opt.timestep)):
    # Prescribe the robot motion only; MuJoCo integrates the box's six free DOFs.
    data.qpos[:box_q] = pose[:box_q]
    data.qpos[0] += 0.25 * data.time
    data.qvel[:robot_dofs] = 0.0
    data.qvel[0] = 0.25
    mujoco.mj_step(model, data)
    for c in data.contact:
      if box not in c.geom or c.dist > 0:
        continue
      other = c.geom[0] if c.geom[1] == box else c.geom[1]
      if other in hands:
        contact_hands.add(model.geom(other).name)
      elif other != floor:
        body_contacts.add(model.geom(other).name)
  return dict(
    delta_xyz_m=(data.qpos[box_q : box_q + 3] - start).tolist(),
    hands=sorted(contact_hands),
    other_robot_contacts=sorted(body_contacts),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mass", type=float, default=2.0)
  args = parser.parse_args()
  push = rollout(args.mass, 0.0)
  miss = rollout(args.mass, 0.95)
  print(
    json.dumps(
      dict(mass_kg=args.mass, prescribed_push=push, offset_control=miss), indent=2
    )
  )
  if (
    len(push["hands"]) != 2
    or push["other_robot_contacts"]
    or push["delta_xyz_m"][0] < 0.04
    or miss["hands"]
    or np.linalg.norm(miss["delta_xyz_m"][:2]) > 0.002
  ):
    raise SystemExit("G1 palm-contact check failed")


if __name__ == "__main__":
  main()
