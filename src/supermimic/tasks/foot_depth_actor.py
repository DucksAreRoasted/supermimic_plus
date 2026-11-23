"""Learned lower-body modulation from measured terrain and future motion poses."""

import hashlib

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from torch import nn
from torch.nn import functional as F


def file_hash(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


class FutureFeet:
  """Tracked ankle position and two orientation axes in the nominal pelvis-yaw frame."""

  def __init__(self, command, *, cache_reference=False):
    self.motion = command.motion
    feet = [
      command.cfg.body_names.index(side + "_ankle_roll_link")
      for side in ("left", "right")
    ]
    positions, rotations, body_chunks = [], [], []
    frames = torch.arange(
      len(self.motion.joint_pos), device=self.motion.joint_pos.device
    )
    for part in frames.split(1024):
      body = self.motion.body_state(part)
      xyz, quat, *_ = body
      if cache_reference:
        body_chunks.append(torch.cat(body, dim=-1))
      positions.append(xyz[:, feet])
      rotations.append(quat[:, feet])
    self.xyz = torch.cat(positions)
    self.quat = torch.cat(rotations)
    self.ahead = frames.new_tensor([4, 8, 12])
    self.axes = self.xyz.new_tensor([[1., 0., 0.], [0., 1., 0.]])
    self._body_table = torch.cat(body_chunks) if cache_reference else None
    self._feet_table = (
      torch.cat([self.compute_feet(part) for part in frames.split(1024)])
      if cache_reference else None
    )

  def body_state(self, time):
    """Cached world-space reference only. The simulated body's FK stays differentiable."""
    if self._body_table is None:
      return self.motion.body_state(time)
    return self._body_table[time].split((3, 4, 3, 3), dim=-1)

  def __call__(self, time):
    if self._feet_table is not None:
      return self._feet_table[time]
    return self.compute_feet(time)

  def compute_feet(self, time):
    ends = self.motion.motion_ends
    last = ends[torch.searchsorted(ends, time, right=True)] - 1
    future = torch.minimum(time[:, None] + self.ahead, last[:, None])
    xyz = self.xyz[future].transpose(1, 2)
    quat = self.quat[future].transpose(1, 2).contiguous()
    yaw = (
      yaw_quat(self.motion.root_quat_w[time])[:, None, None]
      .expand_as(quat)
      .contiguous()
    )
    position = quat_apply_inverse(yaw, xyz - self.motion.root_pos_w[time, None, None])
    axes = [
      quat_apply_inverse(yaw, quat_apply(quat, v.expand_as(xyz)))
      for v in self.axes.unbind()
    ]
    return torch.cat([position, *axes], dim=-1)


class MeasuredPatch(nn.Module):
  """Reproject a measured terrain snapshot into the current (real or imagined) pelvis frame."""

  def __init__(self, base_offsets):
    super().__init__()
    self.register_buffer("base_offsets", base_offsets.clone(), persistent=False)
    axis = torch.arange(25, device=base_offsets.device) * 0.08
    self.register_buffer(
      "crop_xy",
      torch.stack(torch.meshgrid(axis - 0.72, axis - 0.96, indexing="xy"), -1),
      persistent=False,
    )
    self.register_buffer(
      "token_xy",
      torch.cartesian_prod(
        base_offsets.new_tensor([-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]),
        base_offsets.new_tensor([-0.45, -0.30, -0.15, 0, 0.15, 0.30, 0.45]),
      ),
      persistent=False,
    )
    self.register_buffer("sample_xy", torch.cat((self.base_offsets, self.token_xy)), persistent=False)

  def bind_replay(self, row):
    self.bind(row["depth_image"].reshape(-1, 2, 25, 25).float(), row["depth_origin"])

  def capture(self, memory):
    xy = self.crop_xy.expand(len(memory.pos), -1, -1, -1)
    heights, valid = memory.at(xy)
    return torch.stack([heights * 10, valid], 1), torch.cat(
      [memory.pos[:, :2], memory.yaw], -1
    )

  def bind(self, image, origin):
    self.image, self.origin = image, origin

  def sample(self, root, xy):
    offsets = F.pad(xy, (0, 1)).expand(len(root), -1, -1)
    yaw = yaw_quat(root[:, 3:7].contiguous())[:, None].expand(-1, len(xy), -1)
    world_offset = quat_apply(yaw, offsets) + F.pad(
      root[:, None, :2] - self.origin[:, None, :2], (0, 1)
    )
    local = quat_apply_inverse(self.origin[:, None, 2:].expand_as(yaw), world_offset)
    grid = (local[..., :2] - self.crop_xy[0, 0]) / 0.96 - 1
    values = (
      F.grid_sample(self.image, grid.unsqueeze(1), align_corners=True)
      .squeeze(2)
      .transpose(1, 2)
    )
    weighted, support = values.unbind(-1)
    known = (support > 0.25) & (grid.abs() <= 1).all(-1)
    return torch.stack(
      [torch.where(known, weighted / support.clamp_min(0.25), 0), known], -1
    )

  def features(self, root):
    return self.sample(root, self.base_offsets)[..., 0]

  def tokens(self, root):
    samples = self.sample(root, self.token_xy)
    return torch.cat([self.token_xy.expand(len(root), -1, -1), samples], -1)

  def features_and_tokens(self, root):
    features, tokens = self.sample(root, self.sample_xy).split(
      (len(self.base_offsets), len(self.token_xy)), dim=1
    )
    return features[..., 0], torch.cat((self.token_xy.expand(len(root), -1, -1), tokens), -1)


class FootDepthFusion(nn.Module):
  """Two parameter-matched Actor heads. Only the routing of depth tokens differs."""

  def __init__(self, mode, joint_names):
    super().__init__()
    if mode not in ("concat", "query"):
      raise ValueError(mode)
    self.mode = mode
    self.register_buffer(
      "legs",
      torch.tensor(
        [
          [
            joint_names.index(f"{side}_{joint}_joint")
            for joint in (
              "hip_pitch",
              "hip_roll",
              "hip_yaw",
              "knee",
              "ankle_pitch",
              "ankle_roll",
            )
          ]
          for side in ("left", "right")
        ]
      ),
    )
    self.depth = nn.Sequential(nn.Linear(4, 12), nn.SiLU(), nn.Linear(12, 12))
    self.pose = nn.Sequential(nn.Linear(9, 12), nn.SiLU(), nn.Linear(12, 12))
    self.modulator = nn.Sequential(
      nn.Linear(49 * 12 + 3 * 12 + 6, 64), nn.SiLU(), nn.Linear(64, 12)
    )
    nn.init.zeros_(self.modulator[-1].weight)
    nn.init.zeros_(self.modulator[-1].bias)

  def forward(self, action, depth, future):
    valid = depth[..., 3] > 0.5
    tokens = (
      self.depth(
        torch.cat(
          [
            depth[..., :2],
            depth[..., 2:3].masked_fill(~valid[..., None], 0),
            depth[..., 3:],
          ],
          -1,
        )
      )
      * valid[..., None]
    )
    pose = self.pose(future)
    routed = tokens[:, None].expand(-1, 2, -1, -1)
    if self.mode == "query":
      delta = future[..., None, :2] - depth[:, None, None, :, :2]
      score = torch.einsum("blhd,btd->blht", pose, tokens) / (12**0.5)
      score = score - delta.square().sum(-1) / (2 * 0.22**2)
      weights = score.masked_fill(~valid[:, None, None], -1e4).softmax(-1)
      # Keep fixed token order and scale; identical head and inputs for the concat control.
      routed = routed * (weights.mean(2) * valid.sum(-1)[:, None, None])[..., None]
    base = action[:, self.legs]
    features = torch.cat([routed.flatten(2), pose.flatten(2), base], -1)
    scale, shift = self.modulator(features).tanh().chunk(2, -1)
    leg_action = base + 0.15 * (scale * base.tanh() + shift)
    return action.index_copy(1, self.legs.flatten(), leg_action.flatten(1))


def load_fusion(actor, path, actor_path, joint_names, device):
  bundle = torch.load(path, map_location=device, weights_only=True)
  if (
    bundle["base_sha256"] != file_hash(actor_path)
    or list(joint_names) != bundle["joint_names"]
  ):
    raise ValueError("Fusion checkpoint needs its matching base Actor and joint order")
  head = FootDepthFusion(bundle["mode"], list(joint_names)).to(device)
  head.load_state_dict(bundle["state_dict"])
  actor.foot_depth_fusion = head.eval()
  return bundle["mode"]
