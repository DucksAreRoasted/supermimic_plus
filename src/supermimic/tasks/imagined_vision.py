"""Camera observations for state-rendered and latent G1 imagined rollouts."""

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.tasks.depth_mimic import WalkingCameraPattern

CAMERA = WalkingCameraPattern()
PIXELS = CAMERA.height * CAMERA.width
MAX_RANGE = 2.5
LATENT_DIM = 32


def pack_depth(ranges):
  valid = (ranges > 0) & (ranges <= MAX_RANGE) & torch.isfinite(ranges)
  return torch.where(valid, 1 + (ranges / MAX_RANGE * 254).round(), 0).to(torch.uint8)


def unpack_depth(pixels):
  return torch.where(pixels > 0, (pixels.float() - 1) * (MAX_RANGE / 254), -1.0)


def noise_ranges(ranges):
  noisy = torch.where(ranges > 0, ranges + 0.005 * torch.randn_like(ranges), ranges)
  return noisy.masked_fill(torch.rand_like(ranges) < 0.15, -1)


class DepthVision(nn.Module):
  """The WM optimizer owns the image autoencoder and the optional action-conditioned transition."""

  def __init__(self, mode, state_dim, action_dim, *, joint=False, camera_profile="pelvis"):
    super().__init__()
    self.mode = mode
    self.aligned = camera_profile == "instinct"
    self.image_shape = (18,32) if self.aligned else (CAMERA.height,CAMERA.width)
    hidden_h, hidden_w = (3,4) if self.aligned else (4,6)
    if mode not in ("render", "latent"):
      raise ValueError(f"Unknown imagined vision mode {mode}")
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 8, 5, stride=2, padding=2),
      nn.ELU(),
      nn.Conv2d(8, 16, 3, stride=2, padding=1),
      nn.ELU(),
      nn.Conv2d(16, 16, 3, stride=2, padding=1),
      nn.ELU(),
      nn.Flatten(),
      nn.Linear(16 * hidden_h * hidden_w, LATENT_DIM),
      nn.LayerNorm(LATENT_DIM),
    )
    self.decoder = nn.Sequential(
      nn.Linear(LATENT_DIM, 16 * hidden_h * hidden_w),
      nn.ELU(),
      nn.Unflatten(1, (16, hidden_h, hidden_w)),
      nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
      nn.ELU(),
      nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
      nn.ELU(),
      nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
      nn.Sigmoid(),
    )
    self.transition = (
      nn.Sequential(
        nn.Linear(state_dim + action_dim + LATENT_DIM, 128),
        nn.ELU(),
        nn.Linear(128, LATENT_DIM),
      )
      if mode == "latent" and not joint
      else None
    )

    if self.transition is not None:
      nn.init.zeros_(self.transition[-1].weight)
      nn.init.zeros_(self.transition[-1].bias)

  def encode(self, ranges):
    pixels = ranges.float() if self.aligned else pack_depth(ranges).float() / 255
    img = pixels.reshape(-1, 1, *self.image_shape)
    return self.encoder(img).reshape(*ranges.shape[:-1], LATENT_DIM)

  def next_latent(self, z, state, action):
    # Start by persisting the measured feature. Recursive forecasts stay in the
    # layer-normalized coordinates used by the camera encoder.
    delta = self.transition(torch.cat((z, state, action), -1))
    return F.layer_norm(z + delta, (LATENT_DIM,))

  def reconstruction(self, pixels, clean=None):
    # A sampled sequence contains consecutive frames within the same episode.
    z = self.encode(pixels if self.aligned else noise_ranges(unpack_depth(pixels)))
    decoded = self.decoder(z.flatten(0, 1))
    if self.aligned:
      decoded = F.interpolate(decoded, size=self.image_shape, mode="bilinear", align_corners=False)
    recon_loss = F.mse_loss(decoded.reshape_as(pixels), clean.float() if self.aligned else pixels.float()/255)
    return z, recon_loss

  def losses(self, pixels, predicted_states, actions, clean=None):
    z, recon_loss = self.reconstruction(pixels, clean)
    if self.transition is None:
      return recon_loss, None
    current = z[:, 0].detach()
    forecasts = []
    for t in range(predicted_states.shape[1]):
      current = self.next_latent(current, predicted_states[:, t], actions[:, t])
      forecasts.append(current)
    return recon_loss, F.mse_loss(torch.stack(forecasts, 1), z[:, 1:].detach())


class ImaginedCamera:
  """Cast the pelvis sensor's rays in its static group-0 terrain, in a private Warp world."""

  def __init__(self, env, device, batch_size):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    self.device = device
    sim = env.unwrapped.sim
    model = sim.mj_model
    # Packing poses into a single ray batch is valid only when all marked geoms are static.
    visible = model.geom_group == 0
    if (model.body_weldid[model.geom_bodyid[visible]] != 0).any():
      raise ValueError("Packed terrain rendering requires static camera-visible geoms")
    self.model = sim.wp_model
    self.origins, self.directions = CAMERA.generate_rays(model, device)
    self.groups = wp.types.vector(length=6, dtype=float)(-1, 0, 0, 0, 0, 0)
    self.batch_size = batch_size
    with wp.ScopedDevice(device):
      self.data = mjw.put_data(
        model, mujoco.MjData(model), nworld=1, nconmax=1, njmax=1
      )
      self.rc = mjw.create_render_context(
        model,
        nworld=1,
        enabled_geom_groups=[0],
        cam_active=[False] * model.ncam,
        use_textures=False,
        use_shadows=False,
      )
      mjw.kinematics(self.model, self.data)
      mjw.refit_bvh(self.model, self.data, self.rc)
      self.excluded = wp.from_torch(
        torch.full((batch_size * PIXELS,), -1, device=device, dtype=torch.int32)
      )

  @torch.no_grad()
  def ranges(self, root):
    import mujoco_warp as mjw
    import warp as wp
    from mjlab.utils.lab_api.math import matrix_from_quat

    if not 0 < len(root) <= self.batch_size:
      raise ValueError(f"Expected at most {self.batch_size} camera poses, got {len(root)}")
    rotation = matrix_from_quat(root[:, 3:7])
    p = (
      (self.origins @ rotation.transpose(-1, -2) + root[:, None, :3])
      .reshape(1, -1, 3)
      .contiguous()
    )
    v = (self.directions @ rotation.transpose(-1, -2)).reshape_as(p).contiguous()
    dist = torch.empty(p.shape[:2], device=root.device)
    ids = torch.empty_like(dist, dtype=torch.int32)
    normals = torch.empty_like(p)
    with (
      wp.ScopedDevice(self.device),
      wp.ScopedStream(
        wp.stream_from_torch(torch.cuda.current_stream(self.device)), sync_enter=False
      ),
    ):
      mjw.rays(
        m=self.model,
        d=self.data,
        pnt=wp.from_torch(p, dtype=wp.vec3),
        vec=wp.from_torch(v, dtype=wp.vec3),
        geomgroup=self.groups,
        flg_static=True,
        bodyexclude=self.excluded if len(root) == self.batch_size else self.excluded[:len(root) * PIXELS],
        dist=wp.from_torch(dist),
        geomid=wp.from_torch(ids),
        normal=wp.from_torch(normals, dtype=wp.vec3),
        rc=self.rc,
      )
    return dist.view(len(root), PIXELS).masked_fill(
      dist.view(len(root), PIXELS) > MAX_RANGE, -1
    )
