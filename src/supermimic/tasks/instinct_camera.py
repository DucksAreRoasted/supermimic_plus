"""InstinctMJ G1 camera profile (pinned upstream components in third_party)."""

import math

import torch
import warp as wp
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, matrix_from_quat

from supermimic.third_party.instinct_camera.depth import DepthPipeline
from supermimic.third_party.instinct_camera.grouped_ray_caster_camera import (
  GroupedRayCasterCamera,
)
from supermimic.third_party.instinct_camera.grouped_ray_caster_camera_cfg import (
  GroupedRayCasterCameraCfg,
)
from supermimic.third_party.instinct_camera.render import SceneRays
from supermimic.third_party.instinct_camera.rotations import mm, mv

PITCH = math.radians(48)
OFFSET = (
  0.04764571478 + 0.0039635 - 0.0042 * math.cos(PITCH),
  0.015,
  0.46268178553 - 0.044 + 0.0042 * math.sin(PITCH) + 0.016,
)
ROTATION = (
  math.cos(math.radians(0.25)) * math.cos(PITCH / 2),
  math.sin(math.radians(0.25)),
  math.sin(PITCH / 2),
  0.0,
)


class TorsoCamera(GroupedRayCasterCamera):
  def initialize(self, mj_model, model, data, device):
    super().initialize(mj_model, model, data, device)
    limits = self.ray_starts.new_tensor(
      [0.01, 0.01, 0.01, math.radians(2), math.radians(10), math.radians(2)]
    )
    errors = (torch.rand(data.nworld, 6, device=device) * 2 - 1) * limits
    self.ray_starts += errors[:, None, :3]
    rotation = quat_from_euler_xyz(*errors[:, 3:].unbind(-1))
    self.ray_directions = quat_apply(
      rotation[:, None].expand(-1, self.num_rays, -1), self.ray_directions
    )
    self.pipeline = DepthPipeline(device)
    self.noisy_image = torch.zeros(data.nworld, 576, device=device)

  def postprocess_rays(self):
    super().postprocess_rays()
    raw = self._camera_data.output["distance_to_image_plane"]
    fresh = self.pipeline.noised(raw).flatten(1)
    self.noisy_image = torch.where(self.refresh_mask[:, None], fresh, self.noisy_image)

  def observe(self, memory, root):
    data = self.data
    plane = data.output["distance_to_image_plane"].flatten(1)
    distance = plane / self.ray_directions[..., 0]
    local = self.ray_starts + self.ray_directions * distance[..., None]
    points = data.pos_w[:, None] + quat_apply(
      data.quat_w_world[:, None].expand(-1, self.num_rays, -1), local
    )
    memory.update_cloud(points, (plane > 0) & (plane <= 2), root[:, :3], root[:, 3:7])
    return self.noisy_image, self.pipeline.clean(plane.view(-1, 27, 48, 1)).flatten(1)


def camera_config():
  return GroupedRayCasterCameraCfg(
    name="walk_depth",
    class_type=TorsoCamera,
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    pattern=PinholeCameraPatternCfg(height=27, width=48, fovy=58.0),
    focal_length=1.0,
    horizontal_aperture=2 * math.tan(math.radians(87) / 2),
    vertical_aperture=2 * math.tan(math.radians(58) / 2),
    offset=GroupedRayCasterCameraCfg.OffsetCfg(
      pos=OFFSET, rot=ROTATION, convention="world"
    ),
    include_geom_groups=(0, 2),
    exclude_parent_body=False,
    ray_alignment="base",
    mesh_filter_max_hops=24,
    depth_clipping_behavior="max",
    min_distance=0.05,
    max_distance=1e6,
    update_period=1 / 60,
  )


class TorsoRenderer:
  """Cast against posed robot geometry; both the torso and the limbs come from imagined qpos."""

  def __init__(self, env, capacity, device):
    self.cam = env.scene["walk_depth"]
    self.scene = SceneRays(env, min(capacity, 128), True, device, camera=self.cam)
    self.torso = self.cam._require_attachment_frame_metadata().body_id
    self.off = self.cam.ray_starts.new_tensor(OFFSET)
    self.rot = matrix_from_quat(self.off.new_tensor(ROTATION))
    self.device = device

  @torch.compiler.disable
  @torch.no_grad()
  def raw(self, root, joints, ids):
    scene = self.scene
    world = scene.main
    raw = []
    with wp.ScopedStream(
      wp.stream_from_torch(torch.cuda.current_stream(self.device)), sync_enter=False
    ):
      for start in range(0, len(root), world.nworld):
        sl = slice(start, start + world.nworld)
        q = root[sl]
        n = len(q)
        world.qpos[:n, scene.free_q[:7]] = q[:, :7]
        world.qpos[:n, scene.joint_q] = joints[sl]
        world.update()
        torso_r = wp.to_torch(world.d.xmat)[:n, self.torso].reshape(n, 3, 3)
        torso_p = wp.to_torch(world.d.xpos)[:n, self.torso]
        pos = torso_p + mv(torso_r, self.off)
        rot = mm(torso_r, self.rot)
        selected = ids[sl]
        dirs = self.cam.ray_directions[selected]
        o = pos[:, None] + mv(rot[:, None], self.cam.ray_starts[selected])
        v = mv(rot[:, None], dirs)
        if n < world.nworld:
          o = torch.cat((o, o[-1:].expand(world.nworld - n, -1, -1)))
          v = torch.cat((v, v[-1:].expand(world.nworld - n, -1, -1)))
        dist = scene._first_hits(o.contiguous(), v.contiguous())[0][:n]
        raw.append(torch.where((dist >= 0) & (dist <= 1e6), dist * dirs[..., 0], 1e6))
    return torch.cat(raw).view(-1, 27, 48, 1)

  def image(self, root, state, ids):
    from supermimic.utils.gencoord_helpers import decompose_rep_state

    raw = self.raw(root, decompose_rep_state(state)[4], ids)
    return self.cam.pipeline.noised(raw).flatten(1)
