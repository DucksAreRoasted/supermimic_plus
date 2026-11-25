"""Compare ankle-only and contact-aware sole IK on the nine paired failure windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from eval_g1_ankle_optimization import load_windows
from eval_g1_foothold_synthesis import SynthesisRollout, summarize
from eval_g1_support_root import adapt_window
from eval_scene_flat_regression import evaluate
from probe_g1_terrain_reference import Clearance, collision_scene

from supermimic.tasks.terrain_reference import TerrainLegReference


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("checkpoint", "offline", "protocol", "starts", "output"):
    parser.add_argument(name, type=Path)
  parser.add_argument("--geometry-only", action="store_true")
  args = parser.parse_args()
  torch.set_num_threads(1)
  protocol, clips, originals, report = load_windows(args.offline, args.protocol)
  model, terrain = collision_scene(
    report["terrain_seed"], rough_scale=report["rough_scale"]
  )
  adapter = TerrainLegReference(model, terrain.height_at)
  collision = Clearance(adapter)
  args.output.mkdir(parents=True, exist_ok=True)

  def height(xy):
    return terrain.height_at(torch.as_tensor(xy, dtype=torch.float32)).numpy()

  sequences = {}
  metadata = {}
  geometry = {}
  for arm, multipoint in (("ankle", False), ("sole", True)):
    rows, details, depths = [], [], []
    for source in originals:
      row, info = adapt_window(
        adapter, source["original"], height, multipoint_contact=multipoint
      )
      rows.append(row)
      details.append(info)
      depths.extend(
        collision.contact(pose).max() for pose in row["clearance_optimized"]
      )
    sequences[arm] = rows
    metadata[arm] = details
    geometry[arm] = dict(
      penetration_over_5mm=int(np.count_nonzero(np.asarray(depths) > 0.005)),
      peak_penetration_m=float(max(depths)),
      active_contact_points=sum(x["active_contact_points"] for x in details),
    )
    np.savez_compressed(
      args.output / f"{arm}_references.npz",
      **{
        Path(clip["path"]).stem: row["clearance_optimized"]
        for clip, row in zip(clips, rows, strict=True)
      },
    )

  result = dict(geometry=geometry, metadata=metadata)
  if not args.geometry_only:
    probes = {}
    for arm, rows in sequences.items():
      probe = SynthesisRollout(clips, rows)
      evaluated = evaluate(
        args.checkpoint, protocol, args.starts, "cuda:0",
        rough_seed=report["terrain_seed"], rough_scale=report["rough_scale"],
        max_steps=max(clip["horizon"] for clip in clips), rollout_probe=probe,
      )
      probes[arm] = probe
      (args.output / f"{arm}.json").write_text(
        json.dumps(evaluated, indent=2, allow_nan=False) + "\n"
      )
    result["per_motion"] = summarize(clips, probes)
    result["traces"] = {name: probe.trace for name, probe in probes.items()}
    for arm in probes:
      metrics = [row["arms"][arm] for row in result["per_motion"]]
      geometry[arm]["terminations"] = sum(
        x["termination_step"] is not None for x in metrics
      )
      geometry[arm]["slip_m_s"] = (
        sum(x["slip_distance_m"] for x in metrics)
        / max(sum(x["stance_seconds"] for x in metrics), 1e-9)
      )
  (args.output / "comparison.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n"
  )
  print(json.dumps(geometry, indent=2), flush=True)


if __name__ == "__main__":
  main()
