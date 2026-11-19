"""Sliding real-trajectory window, with explicit auto-reset boundaries."""

from collections import deque

import torch


class SimulatorWindow:
  def __init__(self, horizon: int):
    if horizon < 1:
      raise ValueError("simulator calibration horizon must be positive")
    self.steps: deque = deque(maxlen=horizon)
    self.last_obs: torch.Tensor | None = None

  def append(self, obs, next_obs, reward, reset, terminated, clip_end):
    # The runner builds both critic observations with torch.cat (owned storage);
    # mjlab overwrites the returned reward tensor in place on every env.step.
    reset = reset.detach().to(dtype=torch.bool, copy=True)
    # For a pure timeout the wrapper returns the *new* episode, not the final
    # observation. Censor this one transition instead of inventing a value.
    censored = reset & ~(terminated.bool() | clip_end.bool())
    self.steps.append((obs.detach(), reward.detach().clone(), reset, censored))
    self.last_obs = next_obs.detach()

  @property
  def full(self):
    return len(self.steps) == self.steps.maxlen

  def batch(self):
    if not self.full or self.last_obs is None:
      raise RuntimeError("simulator window is not full")
    obs, rewards, resets, censored = zip(*self.steps, strict=True)
    return (
      torch.stack((*obs, self.last_obs)),
      torch.stack(rewards),
      torch.stack(resets),
      torch.stack(censored),
    )
