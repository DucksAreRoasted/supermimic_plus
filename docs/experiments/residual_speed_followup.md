# Residual correction speed follow-up

The completed R-zero-init run used an independent `(2048, 2048, 2048)` residual MLP. It reached a peak reward of 40.50 and a final logged reward of 39.85 at roughly 9.86k steps/s.

Two capacity reductions were screened on the same configuration. Independent `(512, 512)` and `(1024, 1024)` heads improved throughput to about 14.5k and 13.8k steps/s, but both showed severe early reward degradation. A shared-trunk linear residual head reached about 14.8k steps/s and also degraded early reward.

The evidence indicates that the residual needs an independent nonlinear representation. The next speed experiment should use a low-rank residual MLP, with factorized hidden projections and the same zero-initialized output. The first sweep should compare rank 256 and rank 512 at 1K iterations before any 10K run. Keep the full residual head as the quality control and B as the speed control.
