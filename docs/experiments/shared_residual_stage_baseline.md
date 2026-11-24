# 本阶段基线：Shared residual、无 prior、校准 0、24K 热启动

2026-11-20 选定这份 G1 模型作为本阶段姿态跟踪成果，以及后续创新的母版。官方 SuperMimic 和 Twin-mean 继续作为外部对照。实验名为 `g1-shared-focus04-nograph-reset-safe-simweight0-warm24k-plus16k-seed42`，产物目录时间戳为 `2026-11-20_12-29-41`。

## 模型和训练谱系

- 世界模型使用 Shared residual，在共享特征上叠加残差输出。Actor 关闭 latent prior。
- 使用单 Critic，仿真校准损失权重为 0。沿用该 learner 的更新顺序，先更新 Critic，再从同一起始状态重生成 imagined rollout 更新 Actor。
- G1、LAFAN40、seed 42。从同配置从头训练的第 24,000 轮载入 Actor、世界模型及 Critic，再训练 16,000 轮。优化器、回放池和采样器在热启动时重新建立。
- 训练源码为 SuperMimic `e64e34f`、mjlab `3bad5f4`。最终 actor.pt 的 SHA-256 为 `3a076de3137745d1c39d8fb51463483547771000fb8e0c5659f1d7b1765cb768`。

[最终模型和训练清单](../../logs/SuperMimic-GenCoord-G1/g1-shared-focus04-nograph-reset-safe-simweight0-warm24k-plus16k-seed42/2026-11-20_12-29-41/) · [24K 来源模型校验值和热启动命令](../../logs/analysis/reset_safe_nov20/zero_warm24k_plus16k/lineage.json)

## 验收结果

在固定的 40 个参考动作上，从每个片段第 0 帧做无噪声 MuJoCo 回放，按动作等权统计。模型完成 40/40 个动作，每参考帧回报 0.086819，身体位置误差 2.597 cm，身体朝向误差 0.09338 rad，关节角度 L2 误差 0.44480 rad，关节速度 L2 误差 3.72762 rad/s。九个有效 40K 模型中，这份模型的平均身体位置和朝向误差最低，Twin-mean 的平均回报最高。

后续以这份 checkpoint 和相同起点衡量姿态跟踪是否进步，同时观察全局根轨迹、身体速度、完成率及回报。若从母版追加训练，使用从同一 checkpoint、相同预算继续训练的原配置作为对照，以区分算法改动和额外训练带来的收益。

[逐动作配对结果](../../logs/analysis/reset_safe_nov20/zero_warm24k_plus16k/paired/paired_40.csv) · [九模型同起点评估](../../logs/analysis/all_40k_nov20/mujoco_summary.csv)

## 校准权重和仿真数据的作用

本版本中 β=0 指 Critic 损失里的仿真校准权重。训练器仍采集原仿真器轨迹、更新世界模型、从 replay 取出 imagined rollout 的起始状态，并维护仿真校准窗口。拟合 Critic 时会跳过仿真回报目标和对应的 MSE；两个回归 epoch 均使用 imagined GAE 目标。更新后的 Critic 仍给新生成的 Actor rollout 提供末端价值。原训练段及热启动段的 TensorBoard 标量 `Policy/sim_calibration_loss` 全部为 0。

## 归档与恢复

Git 标签 `stage/shared-residual-cal0-40k-seed42` 指向实际训练的 SuperMimic 提交。恢复包另外保存该提交及 mjlab 的源码快照、两个阶段各自的 Actor / Critic / 世界模型、两段运行清单和日志、40 个参考动作、运行环境版本清单、对照 Actor、配对评估代码、固定起点与逐动作结果。

- 本地包位于 `logs/archives/g1-shared-residual-no-prior-cal0-24k-plus16k-seed42.tar.gz`。
- 服务器持久盘副本位于 `/root/gpufree-data/archives/g1-shared-residual-no-prior-cal0-24k-plus16k-seed42.tar.gz`。
- 两份包的 SHA-256 为 `a3ae77f73f6fd0b8f3139b189f63ea1c7c903f0ab22872c7329f09ed93214c65`。Git 保存定位信息，权重和动作文件保存在恢复包中。

从项目根目录在新的临时目录中恢复并校验：

```bash
name=g1-shared-residual-no-prior-cal0-24k-plus16k-seed42
pack="logs/archives/$name.tar.gz"
echo "a3ae77f73f6fd0b8f3139b189f63ea1c7c903f0ab22872c7329f09ed93214c65  $pack" | sha256sum -c - &&
  dest=$(mktemp -d) && tar -xzf "$pack" -C "$dest" &&
  cd "$dest/$name" && sha256sum --quiet -c SHA256SUMS
```

解包根目录的 `archive.json` 记录 `source_run`（24K）、`final_run`（累计 40K）和 `control_run`。每条路径均可相对解包目录解析。恢复既有模型可直接读取 `final_run`。复用原始命令时，将 `lineage.json` 里原服务器的 Python 和初始化目录替换为本机环境及 `source_run`。原始 checkpoint 保存网络权重；继续训练时会重建优化器、回放池和采样器。
