# Residual correction experiment

Base: beta1, replay ratio 16, single joint simulator/imagined Critic,
two regression passes with fixed targets, regenerated Actor rollout, beta_V=1.

The experiment adds an independently initialized MLP with the same hidden
dimensions as the main dynamics MLP. Prediction is main + 0.1 * residual.
Both networks train jointly from scratch. The existing generalized-coordinate
velocity-increment integration is unchanged. This is NOT a PD, inertia,
Lagrangian, or pretrained dynamics prior. There is no residual warmup.

Control: run_beta1_replay16_epochs2_10k.sh. Treatment:
run_g1_residual_correction_10k.sh. Both use 40 clips, seed 42, 4096 environments,
10K iterations, WM horizon 8, policy/simulator horizon 32, simulator loss weight
0.1. The delta-state-named script is not a distinct architecture.

Validation: scripts/probe_residual_correction.py checks CUDA outputs,
zero-scale equivalence, gradients in both networks and a residual optimizer
update. Full-size training smoke uses the treatment script with max-iterations
5 and log-interval 1 before a separate fresh 10K launch.

The CLI requires --runner.residual-correction True (a bare flag is invalid).
