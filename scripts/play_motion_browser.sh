#!/usr/bin/env bash
set -euo pipefail

# Interactive motion browser for a trained SuperMimic checkpoint. The default
# viewer is the local MuJoCo native window; set VIEWER=viser for the web UI.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

MODEL_DIR="${MODEL_DIR:-outputs/new_server_40k_20261114/official_baseline/logs/SuperMimic-GenCoord-G1/2026-11-13_20-28-24}"
MOTION_GLOB="${MOTION_GLOB:-data/modelscope/LAFAN1_Retargeting_Dataset/npz/*.npz}"
DEVICE="${DEVICE:-cuda:0}"
FRAME_RATE="${FRAME_RATE:-30}"
VIEWER="${VIEWER:-native}"

compgen -G "$MOTION_GLOB" >/dev/null || {
  echo "No motions matched: $MOTION_GLOB" >&2
  exit 1
}
[[ -f "$MODEL_DIR/actor.pt" && -f "$MODEL_DIR/world_model.pt" ]] || {
  echo "Checkpoint is incomplete: $MODEL_DIR" >&2
  exit 1
}

export TORCH_COMPILE_DISABLE=1
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-0}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

exec .venv/bin/python -u src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-G1 \
  --model-dir "$MODEL_DIR" \
  --motion-file "$MOTION_GLOB" \
  --viewer "$VIEWER" \
  --device "$DEVICE" \
  --frame-rate "$FRAME_RATE"
