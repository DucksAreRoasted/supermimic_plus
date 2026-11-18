"""Fail unless a residual smoke contains real, finite calibrated updates."""
import math
import sys
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


root = Path(sys.argv[1])
events = list(root.glob("*/events.out*"))
assert len(events) == 1, events
event = events[0]
reader = EventAccumulator(str(event), size_guidance={"scalars": 0})
reader.Reload()
tags = reader.Tags()["scalars"]
for tag in tags:
  assert all(math.isfinite(v.value) for v in reader.Scalars(tag)), tag
for metric in ("imagined_value_loss", "sim_calibration_loss", "critic_total_loss",
               "critic_grad_norm", "sim_calibration_epochs"):
  matches = [t for t in tags if t.endswith("/" + metric)]
  assert len(matches) == 1, (metric, tags)
  values = [v for v in reader.Scalars(matches[0]) if v.step >= 32]
  assert len(values) >= 5, (metric, len(values))
  assert any(v.value > 0 for v in values), metric
  if metric == "sim_calibration_epochs":
    assert all(v.value == 2 for v in values)
  print(matches[0], "last_step", values[-1].step, "value", values[-1].value)
assert max(v.step for t in tags for v in reader.Scalars(t)) == 63
wm = torch.load(event.parent / "world_model.pt", map_location="cpu", weights_only=False)
assert any("residual_mlp" in k for k in wm["dynamics"]), wm["dynamics"].keys()
for name in ("world_model.pt", "actor.pt", "critic.pt"):
  assert (event.parent / name).stat().st_size > 0
print("PASS: full 64 iterations, finite calibrated updates, residual checkpoint")
