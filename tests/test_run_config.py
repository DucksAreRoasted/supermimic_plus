"""Tests for rebuilding a runner config from a wandb-serialized dict."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearnerCfg,
)
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import BaseRunner, SuperMimicRunnerCfg


def _serialized(cfg: SuperMimicRunnerCfg) -> dict:
  """What wandb hands back: the cfg after a JSON round-trip."""
  return json.loads(json.dumps(asdict(cfg)))


def test_flat_checkpoints_do_not_add_camera_fields_to_replay():
  runner = SimpleNamespace(
    cfg=SuperMimicRunnerCfg(),
    _representation=SimpleNamespace(full_state_dim=71),
    _sim_context=SimpleNamespace(output_dim=33),
    _stores_actor_obs=False,
    _num_actions=29,
  )
  schema = BaseRunner._buffer_schema(runner)
  runner.cfg.keep_iteration_checkpoints = True
  assert BaseRunner._buffer_schema(runner) == schema


@pytest.mark.parametrize(
  "policy_cls", [ActorCriticPolicyLearnerCfg, BpttPolicyLearnerCfg]
)
def test_roundtrip_identity(policy_cls):
  # Non-default sizes: a rebuild that ignored the config would still match on
  # defaults; the tuple field doubles as the list-vs-tuple coercion check.
  cfg = SuperMimicRunnerCfg(
    dynamics_hidden_dims=(128, 128), latent_dim=64, policy=policy_cls()
  )
  rebuilt = load_runner_cfg({"runner_cfg": _serialized(cfg)})
  assert type(rebuilt.policy) is policy_cls
  assert rebuilt == cfg


def test_unknown_keys_ignored():
  raw = _serialized(SuperMimicRunnerCfg())
  raw["removed_field"] = 3
  raw["tasks"]["velocity"]["also_removed"] = 1
  assert load_runner_cfg({"runner_cfg": raw}) == SuperMimicRunnerCfg()


def test_missing_fields_warn_with_dotted_paths(capsys):
  raw = _serialized(SuperMimicRunnerCfg())
  del raw["dynamics_hidden_dims"]
  del raw["tasks"]["velocity"]["window_frames"]
  load_runner_cfg({"runner_cfg": raw})
  out = capsys.readouterr().out
  assert "dynamics_hidden_dims" in out
  assert "tasks.velocity.window_frames" in out


def test_missing_runner_cfg_raises():
  with pytest.raises(ValueError, match="runner_cfg"):
    load_runner_cfg({})
  with pytest.raises(ValueError, match="runner_cfg"):
    load_runner_cfg({"runner_cfg": {}})


def test_representation_must_match_task_id():
  check_representation_matches("SuperMimic-GenCoord-G1", "gencoord")
  check_representation_matches("SuperMimic-MaxCoord-G1", "maxcoord")
  with pytest.raises(ValueError, match="maxcoord"):
    check_representation_matches("SuperMimic-MaxCoord-G1", "gencoord")
  with pytest.raises(ValueError, match="gencoord"):
    check_representation_matches("SuperMimic-GenCoord-G1", "maxcoord")
