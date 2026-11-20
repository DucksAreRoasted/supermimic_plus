#!/usr/bin/env bash
# Warm-start a matched low-step transfer arm on an already retargeted G1 clip.
set -euo pipefail
arm=${1:?expected blind, actor, or actor-world}
case "$arm" in blind|actor|actor-world) ;; *) exit 2 ;; esac
checkpoint=${2:?expected archived shared-residual cal0 checkpoint directory}
motion=${3:?expected adapted walking clip}
iterations=${4:-192}
waypoints=${5:-False}
case "$waypoints" in True) suffix=-waypoints ;; False) suffix= ;; *) exit 2 ;; esac
python_bin=${PYTHON_BIN:-.venv/bin/python}
export WANDB_MODE=${WANDB_MODE:-disabled} PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m supermimic.scripts.train SuperMimic-GenCoord-G1 \
  --motion-file "$motion" --num-envs 256 --device cuda:0 --env.seed 42 \
  --runner.init-model-dir "$checkpoint" --runner.terrain-inputs "$arm" \
  --runner.terrain-height 0.05 --runner.terrain-phase-cells 0 \
  --runner.max-iterations "$iterations" --runner.log-interval 8 --runner.save-interval "$iterations" \
  --runner.run-name "terrain-5cm-${arm}${suffix}-adapted-${iterations}it-seed42" \
  --runner.use-root-waypoints "$waypoints" \
  --runner.critic-variant sim-calibrated --runner.sim-calibration-weight 0 \
  --runner.sim-calibration-epochs 2 --runner.residual-correction True \
  --runner.residual-shared-trunk True --runner.return-sampler.focus-share 0.4 \
  --runner.replay-ratio 2 --runner.wm.num-batches 1 --runner.wm.batch-size 256 \
  --runner.wm.horizon 8 --runner.policy.num-batches 1 --runner.policy.batch-size 256 \
  --runner.policy.horizon 32 --runner.policy.lr 0.00002 --runner.wm.lr 0.00002
