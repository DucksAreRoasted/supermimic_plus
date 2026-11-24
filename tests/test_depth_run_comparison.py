import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
  "compare_depth_runs",
  Path(__file__).resolve().parents[1] / "scripts/compare_depth_runs.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def scenario(finish, trace):
  return dict(
    start=17,
    survived=finish,
    termination_terms=[] if finish else ["anchor_pos"],
    steps=len(trace),
    success=finish,
    last_root_xy_error_m=0.2,
    crossed_height_m=0.05,
    reward_per_planned_step=0.03,
    mean_body_error_cm=np.mean(trace),
    body_error_trace_cm=trace,
  )


def test_common_prefix_failures_and_physical_pairing(tmp_path):
  base, candidate = (tmp_path / name for name in ("base", "candidate"))
  protocol = dict(
    clip_sha256="clip",
    height=0.1,
    lateral_height=0.05,
    landscape_seed=1,
    steps=3,
    depth_noise_m=0.005,
    depth_dropout=0.15,
    perception="depth",
    fusion_mode="concat",
    seed=42,
    solver_iterations=60,
    solver_ls_iterations=60,
    external_joint_compensation=False,
    flat=False,
    disjoint_windows=True,
    initial_max_abs_difference={},
  )
  for d, name, row in (
    (base, "concat", scenario(False, [7.0, 8.0])),
    (candidate, "depth", scenario(True, [7.0, 8.0, 10.0])),
  ):
    path = d / "walk"
    path.mkdir(parents=True)
    np.savez(
      path / "starts.npz", frames=[17], qpos=[1], qvel=[2], context=[3], full_state=[4]
    )
    (path / f"{name}.json").write_text(
      json.dumps({**protocol, "windows_detail": [row]})
    )
  result = module.compare(base, candidate, ["walk"])
  assert result["gained"] == 1 and result["candidate"]["survived"] == 1
  assert result["common_body_delta_cm"] == 0 and result["meets_criteria"]
  assert result["reference"]["failures"] == {"terminated": 1}
  assert result["reference"]["termination_terms"] == {"anchor_pos": 1}
  assert result["reference"]["mean_steps_before_termination"] == 2
  assert result["candidate"]["mean_steps_before_termination"] is None
  assert result["pass_rate_met"] and result["tracking_met"]
  (base / "walk/depth.json").write_bytes((base / "walk/concat.json").read_bytes())
  reversed_result = module.compare(candidate, base, ["walk"], "depth.json")
  assert reversed_result["lost_by_reason"] == {"terminated": 1}
  assert not reversed_result["pass_rate_met"] and reversed_result["tracking_met"]
  np.savez(
    candidate / "walk/starts.npz",
    frames=[17],
    qpos=[2],
    qvel=[2],
    context=[3],
    full_state=[4],
  )
  with pytest.raises(AssertionError):
    module.compare(base, candidate, ["walk"])


@pytest.mark.parametrize("outcomes,expected_panels,confirmed,uncertain", [
  ([True, False], ["primary"], False, True),
  ([False, False], ["primary"], False, False),
  ([True, True, False, True], ["primary", "independent"], False, True),
  ([True, True, True, True], ["primary", "independent"], True, False),
])
def test_repeated_physical_decision(monkeypatch, tmp_path, outcomes, expected_panels, confirmed, uncertain):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  import follow_depth_result as workflow

  decisions = iter(outcomes)
  def one_result():
    passed = next(decisions)
    return {"pass_rate_met": passed, "tracking_met": True, "meets_criteria": passed}

  panels = []
  def repeat(checkpoint, directory, panel, reference, controls):
    panels.append(panel)
    return one_result()

  monkeypatch.setattr(workflow, "repeat_panel", repeat)
  monkeypatch.setattr(workflow, "validate_panel", lambda *args: one_result())
  monkeypatch.setattr(workflow, "evaluate", lambda *args: one_result())
  result = workflow.verify(tmp_path, tmp_path/"main", tmp_path/"independent")
  assert panels == expected_panels and result["confirmed"] == confirmed
  assert result["repeat_inconclusive"] == uncertain


def test_depth_wm_lr_ablation_keeps_other_training_parameters(monkeypatch):
  from dataclasses import asdict

  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  from train_g1_dual_head_depth import scratch_config

  common = dict(iterations=20000, batch=1024, seed=42, root_waypoints=False,
                num_envs=4096, official_backends=True, cache_reference=True)
  base = asdict(scratch_config(**common))
  arm = asdict(scratch_config(**common, wm_lr=8e-5))
  assert base["wm"].pop("lr") == base["policy"]["lr"] == 2e-5
  assert arm["wm"].pop("lr") == 8e-5
  assert arm.pop("run_name") != base.pop("run_name")
  assert arm == base
  for invalid in (0, -1.0, float("inf"), float("nan")):
    with pytest.raises(ValueError, match="learning rate"):
      scratch_config(**common, wm_lr=invalid)


def test_g1_depth_actor_inputs(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  from train_g1_dual_head_depth import scratch_config

  args = (10, 2, 42, False, 4)
  default = scratch_config(*args)
  assert (default.actor_obs, default.history_length) == ("proprio", 10)
  assert default.depth_fusion == "concat"
  for name, obs in (("proprio1", "proprio"), ("state1", "rep_state")):
    cfg = scratch_config(*args, actor_input=name)
    assert (cfg.actor_obs, cfg.history_length) == (obs, 1)
    assert cfg.run_name != default.run_name


def test_learning_rate_run_rejects_different_control_manifest(tmp_path, monkeypatch):
  from copy import deepcopy
  from dataclasses import asdict

  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  from follow_depth_result import check_control_config, check_fast_profile
  from train_g1_dual_head_depth import scratch_config

  base_cfg = asdict(scratch_config(40, 1024, 42, False, 4096, official_backends=True,
                                   cache_reference=True))
  baseline = dict(runner_cfg=base_cfg, motion_files=["walk-a", "walk-b"], seed=42, num_envs=4096,
                  iteration=40, task_id="g1", base_sha256=None, adapter_sha256=None,
                  representation="gencoord", command="train --official-backends")
  control, run = (tmp_path / name for name in ("control", "arm"))
  control.mkdir()
  run.mkdir()
  (control / "run_manifest.json").write_text(json.dumps(baseline))
  arm = deepcopy(baseline)
  arm["runner_cfg"]["wm"]["lr"] = 8e-5
  arm["runner_cfg"]["run_name"] += "-wmlr8e-05"
  candidate = run / "run_manifest.json"
  candidate.write_text(json.dumps(arm))
  check_fast_profile(run, 8e-5)
  check_control_config(run, control)
  with pytest.raises(ValueError, match="speed configuration"):
    check_fast_profile(run)
  arm["motion_files"].reverse()
  candidate.write_text(json.dumps(arm))
  with pytest.raises(ValueError, match="motion_files"):
    check_control_config(run, control)
  arm["motion_files"].reverse()
  arm["runner_cfg"]["policy"]["lr"] = 8e-5
  candidate.write_text(json.dumps(arm))
  with pytest.raises(ValueError, match="Another runner"):
    check_control_config(run, control)

  arm = deepcopy(baseline)
  arm["runner_cfg"]["use_root_waypoints"] = True
  candidate.write_text(json.dumps(arm))
  check_control_config(run, control, root_waypoints=True)
  arm["runner_cfg"]["wm"]["lr"] = 8e-5
  candidate.write_text(json.dumps(arm))
  with pytest.raises(ValueError, match="Another runner"):
    check_control_config(run, control, root_waypoints=True)


def test_waypoint_scratch_initial_weights_actions_and_gradient(monkeypatch):
  import torch

  from supermimic.models import MultiEncoderAE
  from tests.conftest import (
    ACTION_DIM,
    LATENT_DIM,
    OBS_DIM,
    make_actor_obs,
    make_actor_specs,
  )

  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  from train_g1_dual_head_depth import initial_parameter_hash

  kwargs = dict(obs_dim=OBS_DIM, action_dim=ACTION_DIM, task_specs=make_actor_specs(),
                slot_mix={"motion": 0.5, "velocity": 0.5}, latent_dim=LATENT_DIM,
                hidden_dims=(32, 32), latent_bc_weight=1.0, terrain_obs_dim=15, history_length=1)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42)
    control = MultiEncoderAE(**kwargs)
    rng_after_control = torch.random.get_rng_state()
    torch.manual_seed(42)
    waypoint = MultiEncoderAE(**kwargs, use_root_waypoints=True)
    assert torch.equal(torch.random.get_rng_state(), rng_after_control)
    assert initial_parameter_hash(control) == initial_parameter_hash(waypoint, omit_waypoint_columns=True)
    obs = make_actor_obs(16)
    obs["terrain"] = torch.randn(16, 15)
    obs["root_waypoints"] = torch.rand(16, 4)
    actual = waypoint(obs)
    torch.testing.assert_close(actual, control(obs), atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert (waypoint.decoder[0].weight.grad[:, -4:].abs().sum(0) > 0).all()


def test_lafan40_protocol_keeps_order_and_rejects_changed_motions(tmp_path, monkeypatch):
  import hashlib

  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  import train_g1_dual_head_depth as train

  monkeypatch.setattr(train, "ROOT", tmp_path)
  clips = []
  for i in reversed(range(40)):
    p = tmp_path / f"motion-{i}.npz"
    p.write_bytes(str(i).encode())
    clips.append(dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  manifest = tmp_path / "protocol.json"
  manifest.write_text(json.dumps({"clips": clips}))
  assert [p.name for p in train.protocol_clips(manifest)] == [c["path"] for c in clips]
  (tmp_path / clips[7]["path"]).write_bytes(b"changed")
  with pytest.raises(ValueError, match="checksum"):
    train.protocol_clips(manifest)
  manifest.write_text(json.dumps({"clips": [clips[0]] * 40}))
  with pytest.raises(ValueError, match="distinct"):
    train.protocol_clips(manifest)


def test_rough40_results_use_paired_starts_and_planned_frame_coverage(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  from compare_lafan40_terrain import METHODS, compare

  for method in METHODS:
    trials = []
    for i in range(40):
      completed = i > 0 or method == "latent"
      trials.append(dict(motion=f"clip{i}", planned_frames=100, steps=100 if completed else 1,
                         completed=completed, termination_terms=["motion_end" if completed else "anchor_ori"],
                         error_body_pos=.02 if completed else .3, mean_pelvis_xy_error_m=.4,
                         reward_per_reference_frame=.08 if completed else .001))
    payload = dict(starts_sha256="paired", terrain_seed=1701, per_motion=trials)
    (tmp_path / f"{method}-rough.json").write_text(json.dumps(payload))
  rows, summary = compare(tmp_path)
  assert len(rows) == 40 and summary["base"]["common_finishes"] == 39
  assert summary["latent"]["gained"] == ["clip0"]
  assert summary["base"]["mean_reference_coverage"] == pytest.approx(.97525)
  assert summary["base"]["common_body_cm"] == pytest.approx(2)
  payload["starts_sha256"] = "different"
  (tmp_path / "render-rough.json").write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="mismatch"):
    compare(tmp_path)
