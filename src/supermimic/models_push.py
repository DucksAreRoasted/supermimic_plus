"""Contact-conditioned G1 and free-box dynamics for model-based pushing."""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from torch import nn

from supermimic.utils.helpers import quat_integrate


def dense(inp, hidden, out):
  net = nn.Sequential(
    nn.Linear(inp, hidden),
    nn.SiLU(),
    nn.Linear(hidden, hidden),
    nn.SiLU(),
    nn.Linear(hidden, out),
  )
  nn.init.zeros_(net[-1].weight)
  nn.init.zeros_(net[-1].bias)
  return net


def local_vectors(q, vectors):
  return quat_apply_inverse(q.unsqueeze(-2).expand(*vectors.shape[:-1], 4), vectors)


class ContactDynamics(nn.Module):
  def __init__(self, num_state=68, num_action=29, dt=0.02):
    super().__init__()
    self.dt = dt
    self.register_buffer("size", torch.tensor([0.34, 0.27, 0.41]))
    self.ground = dense(12, 64, 6)
    self.push = dense(24, 96, 6)
    self.reaction = dense(24 + num_state + num_action, 96, num_action + 6)
    self.register_buffer("twist_scale", torch.tensor([0.5, 0.5, 0.5, 2.0, 2.0, 2.0]))

  def features(self, box, palms, next_palms, physics):
    q = box[..., 3:7]
    twist = local_vectors(q, box[..., 7:].unflatten(-1, (2, 3))).flatten(-2)
    up = torch.zeros_like(box[..., :3])
    up[..., 2] = 1.0
    gravity_axis = quat_apply_inverse(q, up)
    base = torch.cat(
      [
        (box[..., 2:3] - self.size[2]) * 5,
        gravity_axis,
        twist / self.twist_scale,
        physics,
      ],
      -1,
    )
    rel = local_vectors(q, palms - box[..., None, :3])
    velocity = local_vectors(q, (next_palms - palms) / self.dt - box[..., None, 7:10])
    gap = (rel.abs() - self.size).clamp_min(0.0).norm(dim=-1)
    # Smooth support outside the measured rubber-hand capsule; no contact term at a distance.
    gate = (1 - (gap.min(-1, keepdim=True).values - 0.065).clamp_min(0.0) / 0.15).clamp(
      0.0, 1.0
    )
    return (
      base,
      torch.cat([base, rel.flatten(-2) * 3.0, velocity.flatten(-2)], -1),
      gate,
    )

  def advance_box(self, box, palms, next_palms, physics):
    base, features, gate = self.features(box, palms, next_palms, physics)
    resting = torch.zeros_like(base)
    resting[..., 3] = 1.0
    resting[..., 10:] = physics
    dv = (
      self.ground(base) - self.ground(resting) + gate * self.push(features)
    ) * self.twist_scale
    local = dv.unflatten(-1, (2, 3))
    q = box[..., 3:7].unsqueeze(-2).expand(*local.shape[:-1], 4)
    new_twist = 0.85 * box[..., 7:] + quat_apply(q, local).flatten(-2)
    avg = 0.5 * (new_twist + box[..., 7:])
    p = box[..., :3] + avg[..., :3] * self.dt
    ori = quat_integrate(box[..., 3:7], avg[..., 3:], self.dt)
    return torch.cat([p, ori, new_twist], -1)

  def robot_delta(self, normalized_state, raw, box, palms, nominal_palms, physics):
    _, feat, gate = self.features(box, palms, nominal_palms, physics)
    return gate * self.reaction(torch.cat([feat, normalized_state, raw], -1))

  def step(self, task, aux, rep, action, ctx, box, palms, physics):
    dyn = task.wm.dynamics
    form = task.wm.representation
    delta = dyn.predict_delta(rep, action, aux, ctx)
    nom_aux, nom_rep = form.apply_prediction(aux, rep, delta, dyn.dt)
    pos, rot, _, _ = form.world_frame_bodies(nom_aux, nom_rep)
    nominal_palms = task.palms(pos, rot)
    reaction = self.robot_delta(
      dyn.state_normalizer(rep), action, box, palms, nominal_palms, physics
    )
    new_aux, new_rep = form.apply_prediction(aux, rep, delta + reaction, dyn.dt)
    pos, rot, _, _ = form.world_frame_bodies(new_aux, new_rep)
    new_palms = task.palms(pos, rot)
    new_box = self.advance_box(box, palms, new_palms, physics)
    return new_aux, new_rep, new_box, new_palms


class PushPolicy(nn.Module):
  """Bounded corrections to the archived tracking policy conditioned on box and goal."""

  def __init__(self):
    super().__init__()
    self.net = dense(68 + 13 + 6 + 2, 96, 29)

  def forward(self, rep, aux, box, palms, goal):
    obs = torch.cat(
      [
        rep.clamp(-8.0, 8.0),
        box[..., :3] - aux[..., :3],
        box[..., 3:],
        (palms - box[..., None, :3]).flatten(-2),
        goal - box[..., :2],
      ],
      -1,
    )
    return 0.65 * self.net(obs).tanh()
