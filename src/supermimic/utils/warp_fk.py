"""Fused floating-base forward kinematics for the eager G1 visual rollout."""

from functools import cache

import torch
import warp as wp
from torch.autograd.function import once_differentiable


@cache
def _path_kernel(depth: int):
  # The adjoint needs each intermediate transform on the path. Unrolling at
  # specialization time lets Warp generate its reverse pass for the same path.
  @wp.kernel(enable_backward=True)
  def transform(
    state: wp.array2d(dtype=wp.float32),
    cols: wp.array2d(dtype=wp.int64),
    qpos0: wp.array2d(dtype=wp.float32),
    constants: wp.array3d(dtype=wp.float32),
    result: wp.array3d(dtype=wp.float32),
  ):
    env, body = wp.tid()
    nj = (state.shape[1] - 13) // 2
    pos = wp.vec3(0.0)
    rotation = wp.quat_identity()
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    for i in range(wp.static(depth)):
      col = wp.int32(cols[i, body])
      angle = float(0.0)
      velocity = float(0.0)
      if col < nj:
        angle = state[env, 13 + col]
        velocity = state[env, 13 + nj + col]
      half_angle = (angle - qpos0[i, body]) * 0.5
      c = wp.cos(half_angle)
      s = wp.sin(half_angle)
      # mjlab stores position, axis, pivot, offset quaternion and its
      # axis product in this order. Its quaternion columns are wxyz.
      anchor = pos + wp.quat_rotate(
        rotation,
        wp.vec3(constants[i, body, 0], constants[i, body, 1], constants[i, body, 2]),
      )
      twist = velocity * wp.quat_rotate(
        rotation,
        wp.vec3(constants[i, body, 3], constants[i, body, 4], constants[i, body, 5]),
      )
      turn = wp.quat(
        c * constants[i, body, 10] + s * constants[i, body, 14],
        c * constants[i, body, 11] + s * constants[i, body, 15],
        c * constants[i, body, 12] + s * constants[i, body, 16],
        c * constants[i, body, 9] + s * constants[i, body, 13],
      )
      rotation = rotation * turn
      pos = anchor - wp.quat_rotate(
        rotation,
        wp.vec3(constants[i, body, 6], constants[i, body, 7], constants[i, body, 8]),
      )
      angular = angular + twist
      linear = linear + wp.cross(anchor, twist)

    root = wp.normalize(
      wp.quat(state[env, 4], state[env, 5], state[env, 6], state[env, 3])
    )
    offset = wp.quat_rotate(root, pos)
    pos = wp.vec3(state[env, 0], state[env, 1], state[env, 2]) + offset
    rotation = root * rotation
    angular = wp.vec3(state[env, 10], state[env, 11], state[env, 12]) + wp.quat_rotate(
      root, angular
    )
    linear = (
      wp.vec3(state[env, 7], state[env, 8], state[env, 9])
      + wp.quat_rotate(root, linear)
      + wp.cross(angular, offset)
    )
    result[env, body, 3] = rotation[3]
    for k in range(3):
      result[env, body, k] = pos[k]
      result[env, body, 4 + k] = rotation[k]
      result[env, body, 7 + k] = linear[k]
      result[env, body, 10 + k] = angular[k]

  return transform


class _BodyState(torch.autograd.Function):
  @staticmethod
  def forward(ctx, state, cols, qpos0, constants):
    result = state.new_empty((state.shape[0], cols.shape[1], 13))
    args = [
      wp.from_torch(x, requires_grad=False) for x in (state, cols, qpos0, constants)
    ]
    stream = wp.stream_from_torch(state.device) if state.is_cuda else None
    wp.launch(
      _path_kernel(cols.shape[0]),
      result.shape[:2],
      inputs=args,
      outputs=[wp.from_torch(result)],
      device=args[0].device,
      stream=stream,
    )
    ctx.save_for_backward(state, cols, qpos0, constants, result)
    return result

  @staticmethod
  @once_differentiable
  def backward(ctx, grad):
    state, cols, qpos0, constants, result = ctx.saved_tensors
    grad_state = torch.zeros_like(state)
    # Several tracked bodies contribute to each root or joint derivative.
    args = [
      wp.from_torch(x, requires_grad=False) for x in (state, cols, qpos0, constants)
    ]
    stream = wp.stream_from_torch(state.device) if state.is_cuda else None
    wp.launch(
      _path_kernel(cols.shape[0]),
      result.shape[:2],
      inputs=args,
      outputs=[wp.from_torch(result, requires_grad=False)],
      adjoint=True,
      adj_inputs=[wp.from_torch(grad_state), None, None, None],
      adj_outputs=[wp.from_torch(grad.contiguous(), requires_grad=False)],
      device=args[0].device,
      stream=stream,
    )
    return grad_state, None, None, None


class WarpForwardKinematics:
  """Use the exact path, joint ordering and fixed offsets from mjlab's FK."""

  def __init__(self, fk):
    wp.init()
    if fk.consts.shape != (*fk.col.shape, 17) or fk.qpos0.shape != fk.col.shape:
      raise ValueError("Unsupported mjlab forward-kinematics path layout")
    self.fk = fk
    self.joint_names = fk.joint_names

  def __call__(
    self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel
  ):
    inputs = (root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel)
    if self.fk.consts.dtype != torch.float32 or any(
      x.dtype != torch.float32 for x in inputs
    ):
      return self.fk(
        root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel
      )
    state = torch.cat(inputs, -1)
    flat = state.reshape(-1, state.shape[-1])
    result = _BodyState.apply(flat, self.fk.col, self.fk.qpos0, self.fk.consts)
    return result.reshape(*state.shape[:-1], *result.shape[-2:]).split(
      (3, 4, 3, 3), dim=-1
    )
