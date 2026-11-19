# G1 轻量化 KL prior（kl-lite）

这份源码来自训练服务器以 `0019f2434f44f00d1b38e4a5d1442b6449554201` 为基础的工作区。保留了实际训练模块 `models.py`、`runners/supermimic_runner.py` 以及同工作区的 `mjlab_cuda_graph_env.py`。服务器没有为当时的每次训练提交源码；这里保存的是取回时的源码，另以当次运行的清单和 checkpoint 参数形状交叉核对。当前源码亦支持后来实验的 blend，但下面两个入口都不启用它。

| 设置 | 1K / 5K 共同取值 |
| --- | --- |
| Actor prior | `latent_prior_mode=kl`，`latent_prior_weight=0.001`，`latent_prior_blend=0.0` |
| Prior 网络 | 历史观测末帧输入；线性层维度 `68 → 512 → 256 → 64`（1K 和 5K 的 `actor.pt` 一致）；输出为 32 维均值和 32 维对数方差 |
| 世界模型 / Critic | Shared residual correction / Simulator-Calibrated Critic，仿真校准权重 `0.1` |
| 采样与训练 | `return_sampler.focus_share=0.4`；40 个 LAFAN1 动作，4096 个环境，seed 42，从头训练 |

`1k_manifest.json` 和 `5k_manifest.json` 是原始训练清单。项目根目录下的以下脚本分别重放清单中的命令，动作列表和顺序已核对：

```bash
bash scripts/run_g1_shared_focus04_latent_prior_kl_lite_1k.sh
# 或者，多轮版本：
bash scripts/run_g1_shared_focus04_latent_prior_kl_lite_5k.sh
```

运行需要 CUDA、清单中 `data/modelscope/LAFAN1_Retargeting_Dataset/npz/` 下的 40 个文件，以及仓库旁边的 `../mjlab_research`。训练服务器用的是 `mjlab_research` 的 `3bad5f445c93437db3300f2ab048d3dba0bdf73c`；其两个图捕获相关提交已存入 `mjlab-graph.bundle`。若要重建，可在**用于复现的**相邻 mjlab 克隆中获取这两个提交（需要已有基础提交 `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`）。从 SuperMimic 根目录执行：

```bash
bundle="$PWD/docs/experiments/kl_prior_lite/mjlab-graph.bundle"
git -C ../mjlab_research bundle verify "$bundle"
git -C ../mjlab_research fetch "$bundle" refs/heads/main:refs/heads/codex/kl-prior-lite-dep
# 仅在专用的 mjlab 克隆中切换：
git -C ../mjlab_research switch codex/kl-prior-lite-dep
```

分支只归档源码、命令、实验清单和上述依赖提交；不含模型权重或 TensorBoard 文件。服务器上的原始实验位于 `logs/SuperMimic-GenCoord-G1/g1-shared-focus04-latent-prior-kl-lite-{1k,5k}-seed42/`。
