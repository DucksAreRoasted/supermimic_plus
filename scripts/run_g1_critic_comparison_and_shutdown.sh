#!/usr/bin/env bash
set -euo pipefail

# Run the matched single-critic/twin-critic experiment and power off only after
# both runs exit successfully. A failed training keeps the machine available.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

./scripts/run_g1_critic_comparison.sh

echo "Both critic experiments completed successfully. Shutting down the server."
shutdown -h now
