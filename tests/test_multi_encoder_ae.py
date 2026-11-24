"""Tests for the MultiEncoderAE actor: per-task goal encoders + shared decoder + latent BC."""

from __future__ import annotations

import pytest
import torch

from supermimic.models import MultiEncoderAE
from tests.conftest import (
  ACTION_DIM,
  LATENT_DIM,
  MOTION_GOAL_DIM,
  OBS_DIM,
  VELOCITY_GOAL_DIM,
  make_actor_obs,
  make_actor_specs,
)

BATCH = 16


def _make_actor(
  slot_mix=None,
  history_length: int = 1,
  latent_prior_weight: float = 0.0,
  latent_prior_mode: str = "mse",
  latent_prior_blend: float = 0.0,
  prior_hidden_dims: tuple[int, ...] | None = None,
) -> MultiEncoderAE:
  return MultiEncoderAE(
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    task_specs=make_actor_specs(),
    slot_mix=slot_mix or {"motion": 0.5, "velocity": 0.5},
    latent_dim=LATENT_DIM,
    hidden_dims=(32, 32),
    latent_bc_weight=1.0,
    history_length=history_length,
    latent_prior_weight=latent_prior_weight,
    latent_prior_mode=latent_prior_mode,
    latent_prior_blend=latent_prior_blend,
    prior_hidden_dims=prior_hidden_dims,
  )


class TestDecoderHistory:
  def test_h1_decoder_width(self):
    actor = _make_actor()
    first_linear = next(m for m in actor.decoder if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == LATENT_DIM + OBS_DIM

  def test_history_window_forward_shape(self):
    actor = _make_actor(history_length=4)
    obs_td = make_actor_obs(BATCH, history_length=4)
    assert actor(obs_td).shape == (BATCH, ACTION_DIM)

  def test_decoder_uses_every_history_frame(self):
    """Perturbing the OLDEST frame must move the action — else H>1 is ignored."""
    actor = _make_actor(history_length=4)
    obs_td = make_actor_obs(BATCH, history_length=4)
    obs_td["history"] = torch.randn(BATCH, 4, OBS_DIM)
    a1 = actor(obs_td)
    obs_td2 = obs_td.clone()
    obs_td2["history"][:, 0] += 5.0
    a2 = actor(obs_td2)
    assert not torch.allclose(a1, a2)


def test_pelvis_extension_preserves_shared_weights_and_actions():
  def build(suffix):
    torch.manual_seed(42)
    return MultiEncoderAE(
      obs_dim=93 + suffix, action_dim=29, task_specs=make_actor_specs(),
      slot_mix={"motion": 1., "velocity": 0.}, latent_dim=32, hidden_dims=(64, 64),
      latent_bc_weight=0., history_length=10, terrain_obs_dim=15,
      zero_init_frame_suffix=suffix,
    )

  control, extended = build(0), build(4)
  original = control.decoder[0].weight
  expanded = extended.decoder[0].weight
  frames = expanded[:, 32:-15].reshape(64, 10, 97)
  assert frames[..., 93:].count_nonzero() == 0
  restored = torch.cat((expanded[:, :32], frames[..., :93].flatten(1), expanded[:, -15:]), 1)
  torch.testing.assert_close(restored, original, rtol=0, atol=0)
  obs = make_actor_obs(BATCH, history_length=10)
  obs["history"] = torch.randn(BATCH, 10, 93)
  obs["terrain"] = torch.randn(BATCH, 15)
  action = control(obs)
  obs["history"] = torch.cat((obs["history"], torch.randn(BATCH, 10, 4)), -1)
  output = extended(obs)
  torch.testing.assert_close(output, action)
  output.sum().backward()
  assert extended.decoder[0].weight.grad[:, 32:-15].reshape(64, 10, 97)[..., -4:].abs().sum() > 0


class TestForwardShape:
  def test_output_shape(self):
    actor = _make_actor()
    obs = make_actor_obs(BATCH)
    actions = actor(obs)
    assert actions.shape == (BATCH, ACTION_DIM)

  def test_odd_batch_distributes_correctly(self):
    actor = _make_actor()
    obs = make_actor_obs(15)
    actions = actor(obs)
    assert actions.shape == (15, ACTION_DIM)


class TestSlotRouting:
  def test_motion_slot_isolation(self):
    """Changing velocity slot goals doesn't affect motion slot outputs."""
    actor = _make_actor()
    obs = make_actor_obs(BATCH)
    out_a = actor(obs)

    # Mutate only the velocity-slot rows of goal_velocity.
    obs2 = obs.clone()
    obs2["goal_velocity"][BATCH // 2 :] = torch.randn(BATCH // 2, VELOCITY_GOAL_DIM)
    out_b = actor(obs2)

    # Motion slots [0, BATCH/2) must match; velocity slots may differ.
    assert torch.allclose(out_a[: BATCH // 2], out_b[: BATCH // 2])

  def test_velocity_slot_isolation(self):
    actor = _make_actor()
    obs = make_actor_obs(BATCH)
    out_a = actor(obs)

    obs2 = obs.clone()
    obs2["goal_motion"][: BATCH // 2] = torch.randn(BATCH // 2, MOTION_GOAL_DIM)
    out_b = actor(obs2)

    # Velocity slots [BATCH/2, BATCH) must match.
    assert torch.allclose(out_a[BATCH // 2 :], out_b[BATCH // 2 :])


class TestSingleTaskCollapse:
  def test_motion_only(self):
    """slot_mix={'motion': 1.0} routes all envs through the motion encoder."""
    actor = _make_actor(slot_mix={"motion": 1.0})
    obs = make_actor_obs(BATCH)
    actions = actor(obs)
    assert actions.shape == (BATCH, ACTION_DIM)

  def test_velocity_only(self):
    actor = _make_actor(slot_mix={"velocity": 1.0})
    obs = make_actor_obs(BATCH)
    actions = actor(obs)
    assert actions.shape == (BATCH, ACTION_DIM)

  def test_swap_slot_mix(self):
    """set_slot_mix changes routing without re-initializing weights."""
    actor = _make_actor(slot_mix={"motion": 0.5, "velocity": 0.5})
    actor.set_slot_mix({"motion": 1.0})
    obs = make_actor_obs(BATCH)
    actions = actor(obs)
    assert actions.shape == (BATCH, ACTION_DIM)


class TestLatentBound:
  def test_decoder_input_is_tanh_bounded(self):
    """tanh bounds the latent fed into the decoder."""
    actor = _make_actor(slot_mix={"motion": 1.0})
    obs = make_actor_obs(BATCH)
    # Hook the decoder's input layer to capture its input.
    captured: list[torch.Tensor] = []

    def hook(_module, args):
      captured.append(args[0].detach())

    actor.decoder[0].register_forward_pre_hook(hook)
    actor(obs)
    decoder_in = captured[0]
    z = decoder_in[..., :LATENT_DIM]
    assert (z.abs() <= 1.0 + 1e-6).all()


class TestGradientFlow:
  def test_gradients_flow_to_all_encoders(self):
    actor = _make_actor()
    obs = make_actor_obs(BATCH)
    actions = actor(obs)
    actions.sum().backward()
    for name, p in actor.encoders["motion"].named_parameters():
      assert p.grad is not None, f"motion.{name} got no grad"
      assert not p.grad.isnan().any()
    for name, p in actor.encoders["velocity"].named_parameters():
      assert p.grad is not None, f"velocity.{name} got no grad"
      assert not p.grad.isnan().any()
    for name, p in actor.decoder.named_parameters():
      assert p.grad is not None, f"decoder.{name} got no grad"


class TestLatentPrior:
  def test_prior_hidden_dims_and_stats(self):
    actor = _make_actor(
      slot_mix={"motion": 1.0},
      latent_prior_weight=1.0,
      prior_hidden_dims=(8, 4),
    ).train()
    obs = make_actor_obs(BATCH)
    actor(obs)

    assert actor.prior is not None
    assert actor.prior[0].out_features == 8
    stats = actor.auxiliary_stats()
    assert set(stats) == {"prior_loss", "prior_kl", "bc_loss"}
    assert torch.isfinite(stats["prior_loss"])
    assert torch.equal(stats["prior_kl"], torch.zeros_like(stats["prior_kl"]))
    assert torch.equal(stats["bc_loss"], torch.zeros_like(stats["bc_loss"]))

  def test_prior_disabled_preserves_baseline_head(self):
    actor = _make_actor(slot_mix={"motion": 1.0})
    assert actor.prior is None

  def test_kl_prior_has_finite_loss_and_gradients(self):
    actor = _make_actor(
      slot_mix={"motion": 1.0},
      latent_prior_weight=0.1,
      latent_prior_mode="kl",
      prior_hidden_dims=(8, 4),
    ).train()
    obs = make_actor_obs(BATCH)
    actions = actor(obs)
    stats = actor.auxiliary_stats()
    assert actor.prior is not None
    assert actor.prior[-1].out_features == 2 * LATENT_DIM
    assert torch.isfinite(stats["prior_kl"])
    assert (stats["prior_kl"] > 0).all()
    actor.zero_grad(set_to_none=True)
    actor.auxiliary_loss().mean().backward()
    assert any(p.grad is not None for p in actor.prior.parameters())
    assert any(p.grad is not None for p in actor.encoders["motion"].parameters())

  def test_prior_blend_changes_decoder_latent(self):
    base = _make_actor(slot_mix={"motion": 1.0}, latent_prior_weight=0.1,
      latent_prior_mode="kl", latent_prior_blend=0.0).eval()
    blend = _make_actor(slot_mix={"motion": 1.0}, latent_prior_weight=0.1,
      latent_prior_mode="kl", latent_prior_blend=0.2).eval()
    blend.load_state_dict(base.state_dict(), strict=False)
    obs = make_actor_obs(BATCH)
    base_in = []
    blend_in = []
    base.decoder[0].register_forward_pre_hook(lambda _m, a: base_in.append(a[0].detach()))
    blend.decoder[0].register_forward_pre_hook(lambda _m, a: blend_in.append(a[0].detach()))
    base(obs)
    blend(obs)
    assert not torch.allclose(base_in[0], blend_in[0])


class TestSlotRanges:
  def test_even_split(self):
    actor = _make_actor(slot_mix={"motion": 0.5, "velocity": 0.5})
    assert actor.slot_ranges(100) == [(0, 50), (50, 100)]

  def test_remainder_absorbed_by_last(self):
    actor = _make_actor(slot_mix={"motion": 0.5, "velocity": 0.5})
    assert actor.slot_ranges(101) == [(0, 50), (50, 101)]

  def test_must_sum_to_one(self):
    with pytest.raises(ValueError, match="sum to 1.0"):
      _make_actor(slot_mix={"motion": 0.3, "velocity": 0.3})

  def test_rejects_unknown_task(self):
    with pytest.raises(ValueError, match="unknown tasks"):
      _make_actor(slot_mix={"motion": 1.0, "motoin": 0.0})

  def test_rejects_negative_fraction(self):
    with pytest.raises(ValueError, match="non-negative"):
      _make_actor(slot_mix={"motion": 1.5, "velocity": -0.5})


class TestUpdateNormalization:
  def test_skips_empty_slots(self):
    # slot_mix={"motion": 1.0} leaves the velocity slot empty; feeding its
    # 0-row slice to goal_normalizers["velocity"] would NaN-poison _mean/_std,
    # so the update must skip empty slots.
    actor = _make_actor(slot_mix={"motion": 1.0}).train()
    actor.update_normalization(make_actor_obs(BATCH))
    vel_norm = actor.goal_normalizers["velocity"]
    assert torch.isfinite(vel_norm._mean).all()
    assert torch.isfinite(vel_norm._std).all()
