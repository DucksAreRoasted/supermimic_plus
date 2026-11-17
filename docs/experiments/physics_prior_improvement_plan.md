# World Model 动力学先验改进计划

## 目标

在现有 `普通 World Model + simulator calibration + epochs=2` 的基础上，逐步加入动力学结构和残差表达能力，提高真实数据的利用效率，同时保持每一步实验都可以单独复现和回退。

实验只比较一个增量：相同 G1 数据、seed、环境数量、replay ratio、策略更新和 10K iteration 预算下，逐步启用新的 World Model 或 Critic 组件。

## 当前基线 A0

A0 不重新训练，使用已经完成的 sim-calibrated epochs=2 结果作为参考。

- representation：`gencoord`
- World Model：当前半隐式积分结构
- Critic：`sim-calibrated`
- `sim_calibration_epochs=2`
- `sim_calibration_weight=0.1`
- `sim_calibration_horizon=32`
- `replay_ratio=16`
- seed：42
- 训练预算：10K iterations

A0 的日志、checkpoint 和 TensorBoard 文件必须保留原实验目录，不覆盖历史结果。

## B：增量状态预测

B 使用当前代码中的增量状态语义：网络预测速度和关节速度的变化量，表示层负责半隐式积分，得到下一时刻状态。

代码位置：

- `src/supermimic/representations.py`：`apply_prediction`
- `src/supermimic/models.py`：Dynamics 输出 prediction
- `scripts/run_g1_delta_state_10k.sh`：10K 启动配置

数据流为：

```text
(state_t, action_t, sim_ctx)
        ↓
   Dynamics MLP
        ↓
(Δv, Δω, Δv_joint)
        ↓
半隐式积分与坐标系更新
        ↓
      state_{t+1}
```

验收重点是奖励曲线、World Model loss、FPS 和 rollout 稳定性。B 不增加 residual head，也不改变 Critic 目标。

## D：半隐式模型 + residual correction

D 在 B 的动力学主干后增加独立 residual head。主干负责可解释的动力学变化量，residual head 学习积分器、接触和未建模效应造成的误差。

当前实现：

- `Dynamics.mlp` 产生主预测；
- `Dynamics.residual_mlp` 产生同维度修正；
- 最终预测为 `base_prediction + residual_scale * residual_prediction`；
- `residual_correction=False` 时完全保持 B 的网络结构和行为；
- D 使用 `--runner.residual-correction --runner.residual-scale 0.1`；
- 启动脚本：`scripts/run_g1_residual_correction_10k.sh`。

首轮实验固定 `residual_scale=0.1`。如果 D 比 B 明显不稳定，再单独测试 0.05 和 0.2；不在同一轮实验中同时改变 loss 权重或 replay 参数。

## E：D + simulator-calibrated Critic

E 在 D 的 World Model 上保留 simulator-calibrated Critic：用真实仿真器窗口计算 value target，并和 imagined rollout 的 Critic loss 联合训练。

- simulator window：32 steps；
- `sim_calibration_lambda=1.0`；
- `sim_calibration_weight=0.1`；
- `sim_calibration_epochs=2`；
- Actor 目标和 beta-v 设定保持不变；
- 只允许 Critic 使用真实仿真器状态、奖励和终止标记。

E 的主要问题不是单步奖励，而是长期 imagined rollout 的价值估计是否更稳定。因此除了 Mean reward，还要记录 Critic loss、sim calibration loss、value disagreement 和 episode length。

## 统一训练协议

每个版本都使用相同的：

- 40 个 LAFAN motion 文件；
- `num_envs=4096`、`device=cuda:0`、`seed=42`；
- `replay_ratio=16`；
- World Model：10 batches、batch size 2048、horizon 8；
- Policy：1 batch、batch size 4096、horizon 32；
- `log_interval=10`、`save_interval=1000`；
- 10K iterations。

每个版本使用独立的 run name 和目录，目录中必须包含 `run_manifest.json`、`actor.pt`、`critic.pt`、`world_model.pt` 和 TensorBoard event 文件。

## 对比方法

对比表固定记录 1K、2K、4K、6K、8K、10K 的：

1. `Train/mean_reward`；
2. `Policy/reward`；
3. `Perf/total_fps`；
4. Mean episode length；
5. World Model state loss；
6. Critic total loss 和 simulator calibration loss；
7. 显存峰值。

最终判断优先看 8K 和 10K 的奖励、达到相同奖励所需的 iteration，以及 FPS。单个早期 checkpoint 的偶然高点不能单独作为结论。

## 代码质量与停止条件

每加入一个组件都必须完成：

- `python -m compileall`；
- 相关单元测试；
- CPU 或小 batch smoke test；
- 检查默认配置与 A0 的 state dict 兼容性；
- 确认关闭新开关时参数量、输出维度和日志字段不变。

如果 residual head 导致 NaN、显存超过可用容量，或 1K–2K 奖励明显低于 B，则停止长跑，保留失败日志，并先调小 residual scale 或恢复到 B。

## 执行顺序

1. A0：复用已有结果，不重新训练。
2. B：当前增量状态预测 10K，完成后记录最终奖励和速度。
3. D：开启 residual correction，完成 10K 并与 B 对比。
4. E：在 D 基础上确认 simulator-calibrated Critic，完成 10K 并与 A0/B/D 对比。
5. 汇总结果，选择最终模型和下一轮需要调参的单一变量。

