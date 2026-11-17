#!/usr/bin/env bash
set -euo pipefail

uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk1_subject1.npz \
  --num-envs 16 --device cuda:0 --env.seed 7 \
  --runner.max-iterations 40 --runner.log-interval 1 --runner.save-interval 40 \
  --runner.replay-ratio 16 \
  --runner.wm.num-batches 1 --runner.wm.batch-size 16 --runner.wm.horizon 2 \
  --runner.policy.num-batches 1 --runner.policy.batch-size 16 --runner.policy.horizon 4 \
  --runner.critic-variant sim-calibrated \
  --runner.sim-calibration-weight 0.1 --runner.sim-calibration-horizon 4 \
  --runner.sim-calibration-interval 1 --runner.sim-calibration-epochs 2 \
  --runner.reference-residual-policy True --runner.reference-residual-scale 0.25 \
  --runner.run-name reference-residual-policy-gpu-smoke-40it-seed7 "$@"
