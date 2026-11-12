"""Protect the official learner and the actual experiment entry point."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from supermimic.learners.policy_learner import ActorCriticPolicyLearner
from supermimic.learners.twin_policy_learner import TwinActorCriticPolicyLearner
from supermimic.runners import supermimic_runner as runner_module
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from tests.test_policy_learner import _ConstantCritic, _DummyActor, _TwoStepEnv

ROOT = Path(__file__).resolve().parents[1]


def test_official_learner_matches_pinned_upstream_bytes():
  # Upstream 5efbf114e11f725c31d83e217be987c13c0e4ce2, before our critic edits.
  source = ROOT / "src/supermimic/learners/policy_learner.py"
  assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "58b29adc8495685fc4dbdff5ecf7b3bcaaa1601689c4e3ba4074cadda48dbd2f"
  )


@pytest.mark.parametrize("variant,expected_count", [("official", 1), ("twin", 2)])
def test_runner_selects_official_or_experimental_without_extra_critic(
  monkeypatch, variant, expected_count
):
  runner = SuperMimicRunner.__new__(SuperMimicRunner)
  runner.cfg = SuperMimicRunnerCfg(critic_variant=variant)
  runner.device = "cpu"
  from unittest.mock import MagicMock

  runner.env = MagicMock()
  runner.env.num_envs = 2
  runner._wm = MagicMock()
  runner._dynamics = MagicMock()
  runner._representation = MagicMock()
  runner.replay_buffer = MagicMock()
  actor = _DummyActor()
  actor.task_specs = []
  actor.slot_ranges = lambda n: []
  monkeypatch.setattr(runner_module, "build_actor", lambda *args: actor)
  monkeypatch.setattr(runner, "_policy_input_dims", lambda: (1, 1))
  monkeypatch.setattr(runner, "_configure_reset_sampling", lambda: None)
  monkeypatch.setattr(
    runner_module, "MultiTaskWorldModelEnv", lambda **kw: _TwoStepEnv()
  )
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  created = []

  def build_critic(*args, **kwargs):
    critic = _ConstantCritic(1.0)
    created.append(critic)
    return critic

  monkeypatch.setattr(runner_module, "build_mlp", build_critic)
  runner._init_policy()
  assert len(created) == expected_count
  expected = (
    ActorCriticPolicyLearner if variant == "official" else TwinActorCriticPolicyLearner
  )
  assert type(runner.policy_learner) is expected
  stats = runner.policy_learner.update()
  assert torch.isfinite(stats["Policy/value_function"])


def test_comparison_launcher_uses_official_recipe_and_one_shared_motion_glob(tmp_path):
  clips = tmp_path / "data/npz"
  clips.mkdir(parents=True)
  for i in range(40):
    (clips / f"clip_{i}.npz").touch()
  binary = tmp_path / "bin"
  binary.mkdir()
  fake_uv = binary / "uv"
  fake_uv.write_text(
    "#!/usr/bin/env python3\n"
    "import json,os,sys\n"
    "with open(os.environ['CAPTURE_ARGS'],'a') as f:\n"
    "    f.write(json.dumps(sys.argv[1:])+'\\n')\n"
  )
  fake_uv.chmod(0o755)
  capture = tmp_path / "arguments.jsonl"
  env = os.environ | {
    "PATH": f"{binary}:{os.environ['PATH']}",
    "CAPTURE_ARGS": str(capture),
    "DATA_ROOT": str(clips.parent),
    "RUN_ROOT": str(tmp_path / "logs"),
  }
  for key in ("CLIP", "MAX_ITERATIONS", "NUM_ENVS", "DEVICE", "SEED"):
    env.pop(key, None)
  subprocess.run(
    ["bash", str(ROOT / "scripts/run_g1_critic_comparison.sh")],
    env=env,
    cwd=tmp_path,
    check=True,
    capture_output=True,
  )
  runs = [json.loads(line) for line in capture.read_text().splitlines()]
  assert len(runs) == 2
  for variant, args in zip(("official", "twin"), runs, strict=True):
    options = dict(zip(args[3::2], args[4::2], strict=True))
    assert options["--motion-file"] == str(clips / "*.npz")
    assert options["--runner.critic-variant"] == variant
    assert options["--num-envs"] == "4096"
    assert options["--env.seed"] == "42"
    assert options["--runner.max-iterations"] == "1000000"
    assert options["--runner.wm.num-batches"] == "10"
    assert options["--runner.wm.batch-size"] == "2048"
    assert options["--runner.wm.horizon"] == "8"
    assert options["--runner.policy.batch-size"] == "4096"
    assert options["--runner.policy.horizon"] == "32"
    assert options["--runner.replay-ratio"] == "16"
