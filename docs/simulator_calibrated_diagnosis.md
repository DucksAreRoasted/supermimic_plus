# 联合 Simulator-Calibrated Critic 诊断与修改方案

## 目的

解释当前联合 Simulator-Calibrated Critic 为什么明显低于官方 baseline 和 Twin-mean，并建立一套可复现的单变量诊断流程。当前结果不能直接归因于 Critic，因为这次实验同时启用了 reference-residual policy、CUDA Graph collection、真实仿真校准和不同的 Critic 更新流程。

当前 10K 结果的代表性数据如下：

| 实验 | Mean reward | Policy/reward | State loss |
|---|---:|---:|---:|
| Simulator-Calibrated 10K | 32.18 | 3.40 | 0.0935 |
| 官方 baseline 40K | 43.39 | 14.37 | 0.0086 |
| Twin-mean 40K | 43.16 | 14.35 | 0.0085 |

`Policy/reward` 是世界模型 imagined rollout 的 Actor 目标，`Mean reward` 是真实环境 episode 回报，二者不能混用。当前两者明显解耦，说明主要问题在 imagined value、Critic 梯度和 World Model 误差之间的关系。

## 已确认的主要风险

### 1. 实验不是严格的单变量对照

当前实验额外启用了：

```text
reference_residual_policy=True
cuda_graph_collection=True
critic_variant=sim-calibrated
```

官方 baseline 和 Twin-mean 使用原始 Actor。因此当前实验同时改变了 Actor、数据分布、采集实现和 Critic，不能把性能差距全部归因于 Simulator-Calibrated Critic。

首轮诊断必须关闭：

```text
--runner.reference-residual-policy False
--runner.cuda-graph-collection False
```

只保留 Simulator-Calibrated Critic。

### 2. Simulator 和 imagined rollout 的动作分布不一致

真实环境采集时，Runner 会加入动作噪声：

```python
actions = actor(actor_td)
actions = actions + torch.randn_like(actions) * policy_cfg.noise
```

Simulator calibration 使用了带噪声策略产生的轨迹，而 imagined Actor rollout 没有加入同样的噪声。因此 Critic 同时拟合：

\[
V^{\pi+\epsilon}_{\mathrm{sim}}
\quad\text{和}\quad
V^{\pi}_{\mathrm{imagined}}
\]

Actor 最终得到的终端价值梯度对应另一个策略分布。

第一轮验证应令 calibration 实验的采集噪声为 0；更完整的方案是让 imagined rollout 使用与真实采集一致的 reparameterized noise。

### 3. Critic 更新强度与官方实现不一致

官方 Critic 使用约 `5 epochs × 4 minibatches` 的更新预算。当前联合 Critic 配置为 `sim_calibration_epochs=2`，每轮只有两次 full-batch 优化。

这会造成：

- imagined target 没有充分拟合；
- simulator target 和 imagined target 同时改变 Critic；
- Critic 在每轮更新后仍然偏离两种 target。

联合版本需要保持官方的 minibatch 数量、epoch 数量、学习率和梯度裁剪规则，再单独加入 simulator loss。

### 4. beta_V warmup 当前没有真正生效

当前实现中的 `_beta_v()` 始终返回 `1.0`。因此训练早期 Critic 尚未可靠时，Actor 已经完全使用其终端价值。

建议使用：

\[
\beta_V(k)=\min\left(1,\frac{k}{K_{\mathrm{warmup}}}\right)
\]

初始建议：

```text
beta_v_start = 0.0 或 0.25
beta_v_end = 1.0
beta_v_warmup = 2000~4000 policy updates
```

`sim_calibration_weight` 也应从 0 逐步升至 0.1，而不是从第一轮固定启用。

### 5. Simulator target 仍然包含 Critic 自举误差

当前 simulator target 使用真实奖励，但末端 value 仍由当前 Critic 提供：

```python
sim_values = critic(sim_obs[:-1])
sim_last_value = critic(sim_obs[-1])
sim_returns = gae_returns(...)
```

因此它不是完全独立的真实仿真回报，而是：

\[
G_{\mathrm{sim}}
=
\text{真实仿真奖励}
+
\gamma^H V_\psi(s_H)
\]

首版可以保留 detached bootstrap，但必须记录 target 的统计量；后续应比较固定 detached target critic 与纯 32 步 Monte Carlo target。

## 诊断指标

每次 policy update 记录以下指标：

```text
Policy/reward
Mean reward
Loss/state
Policy/imagined_value_loss
Policy/sim_calibration_loss
Policy/critic_total_loss
critic_value_mean_imagined
critic_value_mean_sim
imagined_target_mean
sim_target_mean
critic_value_bias
terminal_value_gradient_norm
terminal_gradient_ratio
```

其中：

\[
\text{critic\_value\_bias}
=
V_\psi(s)-G_{\mathrm{target}}
\]

\[
\text{terminal\_gradient\_ratio}
=
\frac{\|\nabla_\theta J_{\mathrm{terminal}}\|}
{\|\nabla_\theta J_{\mathrm{reward}}\|+\epsilon}
\]

这些指标用于判断下降来自 target 偏差、Critic 拟合不足，还是 Actor 终端梯度方向发生改变。

## 第一阶段实验矩阵

所有实验使用相同 seed、40 个动作、World Model 配置、Actor 配置、replay ratio、batch size、horizon 和训练轮数。

| 实验 | reference residual | Simulator calibration | beta_V | 目的 |
|---|---|---|---|---|
| A | 关闭 | 关闭 | 1.0 | 官方单 Critic 对照 |
| B | 关闭 | 开启 | 0→1 warmup | 只测试联合 Critic |
| C | 开启 | 关闭 | 1.0 | 只测试 reference-residual |
| D | 开启 | 开启 | 0→1 warmup | 测试二者组合 |

首轮每个实验跑 5K，确认趋势后再决定是否跑 40K。

### 第一阶段的固定配置

```text
reference_residual_policy = False（A/B）
cuda_graph_collection = False
sim_calibration_horizon = 32
sim_calibration_lambda = 1.0
sim_calibration_weight = 0→0.1 warmup
beta_V = 0→1 warmup
critic update = 官方 5 epochs × 4 minibatches
MSE loss
官方 value clamp
```

## 代码修改顺序

### 第一步：恢复严格对照

在 Runner 中确保 Simulator-Calibrated 实验可以独立关闭 reference-residual 和 CUDA Graph。实验脚本必须把最终解析后的配置写入 `run_manifest.json`。

### 第二步：实现 beta_V 和 calibration weight warmup

在 learner 中按 policy update 次数计算 warmup 系数，并记录：

```text
Policy/beta_v
Policy/sim_weight
```

warmup 不应依赖环境 collection step，避免 replay ratio 改变实际曲线。

### 第三步：对齐 Critic 更新预算

将 imagined 和 simulator target 放入统一的固定 batch，使用与官方一致的 epoch、minibatch、学习率和 gradient clipping。target 在 optimizer step 前生成并 detach，在整个 update 周期内保持不变。

### 第四步：对齐动作噪声

先提供一个无噪声 calibration 模式验证因果关系。之后再实现固定随机噪声或 reparameterized noise，使 simulator rollout 和 imagined rollout 对应同一个策略分布。

### 第五步：比较两类 simulator target

实现两个可切换模式：

1. `detached_critic_bootstrap`：真实奖励加 detached target critic bootstrap；
2. `mc32`：纯 32 步仿真回报，末端不使用当前 Critic。

只有在 B 实验稳定后才比较这两个 target。

## 判定标准

一个修改只有同时满足以下条件，才进入 40K：

- GPU smoke 连续 5 轮无 NaN、无 graph capture 错误；
- imagined value loss 和 sim calibration loss 不持续发散；
- `critic_value_mean` 与两类 target 的差距保持稳定；
- `terminal_gradient_ratio` 没有持续增大到压制即时 reward 梯度；
- 5K 时 Mean reward 不低于严格 baseline 的 95%；
- State loss 没有显著高于同阶段 baseline；
- 实验配置和 checkpoint 可复现加载。

## 当前结论

当前结果不能证明 Simulator-Calibrated Critic 本身无效。更准确的结论是：当前版本把 Critic、Actor、动作噪声、reference-residual、World Model 数据分布和 CUDA Graph collection 同时改动了，导致联合目标无法稳定传递到 Actor。

下一步应先做严格单变量对照，再处理动作分布和 beta_V warmup，最后才比较不同 simulator target。暂不建议增加 Critic 数量、提高 calibration 权重或引入更复杂的物理先验。
