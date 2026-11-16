# Early joint Critic + world-model loss experiment

Historical reference: 91e624f (joint imagined/simulator Critic loss with gradient fix).
The 2026-11-14_21-12-17 manifest has no commit hash, so this is a history-based reconstruction, not a verified byte-identical run snapshot.

Restored: one imagined rollout, Actor update first, joint Critic update second; no beta scaling, no residual actor, no second imagined rollout. Core learner, representation and train entry point restored to the reference.
Retained: world-model semi-implicit/normalized acceleration losses and the read-only state probe.

Launch parameters: scripts/run_early_joint_physics.sh copies the historical manifest including all 40 explicit motion paths and replay_ratio=16. Later beta1/physics runs used replay_ratio=4; iteration-matched rewards do not control for update budget.

Validation: Python compileall only. Not launched or deployed. Historical CUDA capture with dynamically replaced simulator batches needs validation across capture/replay boundaries before a long run. Existing physical-loss nan_to_num guards and missing 1000-iteration warmup also require review before claiming physical validation.

The current remote physics run is unchanged and must retain its original experimental label.
