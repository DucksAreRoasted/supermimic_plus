#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/play_g1_40k.sh [twin|single] [clip] [play.py options...]
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
variant="${1:-twin}"
if (( $# > 0 )); then shift; fi
clip="${1:-walk1_subject1}"
if (( $# > 0 )); then shift; fi

case "$variant" in
  twin) run="2026-11-12_15-43-29-twin" ;;
  single) run="2026-11-12_12-53-47-single" ;;
  *) echo "Usage: $0 [twin|single] [clip] [play.py options...]" >&2; exit 2 ;;
esac

# Inference does not need training-time compilation or its first-frame delay.
export TORCH_COMPILE_DISABLE=1
# Compatible GLX display on this desktop; CUDA physics/inference stay on the GPU.
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec .venv/bin/python -u src/supermimic/scripts/play.py SuperMimic-GenCoord-G1 \
  --model-dir "logs/SuperMimic-GenCoord-G1/$run" \
  --motion-file "outputs/server_40k_20261113/raw/data/modelscope/LAFAN1_Retargeting_Dataset/npz/$clip.npz" \
  --viewer native --device cuda:0 "$@"
