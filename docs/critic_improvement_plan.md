# SuperMimic Critic 改进方案

## 目标

当前 SuperMimic 的 critic 主要是一个基于 privileged observation 的普通 MLP，用 rollout return 做回归，并在 actor 更新后单独拟合。这个设计可以工作，但在 world model 持续变化、imagined rollout 较多的情况下，容易出现 value 高估、对分布外状态过度自信，以及长时序回报方差过大的问题。

本方案的目标是让 critic：

- 更稳定地估计长时序回报；
- 识别 world model 不可靠的 imagined state；
- 降低 actor 利用 world model 误差的风险；
- 为后续使用 SAC 或其他 actor 更新方式提供更可靠的 value 信号。

## 一、Twin Critic：降低 value 高估

将单个 critic 改为两个参数独立的 critic：

\[
V_{\theta_1}(s), \quad V_{\theta_2}(s)
\]

训练时两个网络都回归同一个 target，actor 更新和 advantage 计算使用保守值：

\[
V_{\mathrm{safe}}(s)=\min(V_{\theta_1}(s),V_{\theta_2}(s))
\]

对于基于 action 的目标，也可以使用 TD3/SAC 风格的 clipped minimum：

\[
y_t=r_t+\gamma(1-d_t)\min(Q_{\theta_1}(s_{t+1},a_{t+1}),Q_{\theta_2}(s_{t+1},a_{t+1}))
\]

在当前 PPO critic 上，最小可行改动是：

- 保留现有 critic 输入和训练流程；
- 构造两个结构相同、参数独立的 value network；
- advantage 使用 `min(V1, V2)`；
- 两个 critic 的 loss 取平均；
- 记录 `V1 - V2` 的均值和方差。

这样可以优先解决 world model 早期不准确时 critic 对高回报的过度估计问题。

## 二、Distributional Critic：从标量 value 改为回报分布

当前 critic 只估计期望回报：

\[
V(s)=E[G_t\mid s]
\]

但 motion tracking 的回报可能因为接触失败、终止概率、姿态误差和动作平滑度而呈现较大方差，甚至具有多峰特征。可以让 critic 预测多个 return quantile：

\[
\{q_{\tau}(s)\}_{\tau\in\mathcal{T}}
\]

建议首先尝试 8～16 个 quantiles，并使用 quantile Huber loss 训练。actor 可以使用：

- 中位数作为较稳健的 value；或
- lower quantile 作为保守 value；或
- 根据训练阶段在保守 value 和均值之间逐步切换。

Distributional critic 还能直接提供 return uncertainty，用于判断 imagined rollout 是否可靠。

## 三、把 world model 不确定性加入 critic 输入

当前 critic 使用完整 world model state，但没有显式知道这个状态是否处于 world model 熟悉的分布内。建议向 critic observation 增加：

- world model prediction error 的 EMA；
- dynamics ensemble disagreement；
- latent state 的 visitation density；
- 当前 rollout depth；
- reward model 的置信度；
- 当前状态相对数据集分布的距离。

可以构造如下输入：

```text
critic_input = [
    privileged_state,
    wm_prediction_error_ema,
    ensemble_disagreement,
    rollout_depth,
    state_density,
]
```

这样 critic 不仅估计“这个状态可能有多高的回报”，还能够判断“自己对这个估计有多大把握”。

## 四、Value Ensemble：显式估计不确定性

如果计算资源允许，可以使用 5 个较小的 critic，而不是 2 个较大的 critic：

\[
\bar V(s)=\mathrm{mean}_i V_i(s),\qquad
\sigma_V(s)=\mathrm{std}_i V_i(s)
\]

actor 使用带不确定性惩罚的保守 value：

\[
V_{\mathrm{safe}}(s)=\bar V(s)-\beta\sigma_V(s)
\]

其中 \(\beta\) 在训练早期取较大值，随着 world model 稳定逐步衰减。

对应的 conservative advantage 可以写成：

\[
A_{\mathrm{safe}}=\bar V-\beta\sigma_V
\]

这个机制能够主动抑制 actor 进入 critic 不确定、world model 可能失真的状态区域。

## 五、避免硬 value clamp

当前 critic 输出存在类似：

```python
self.critic(obs["critic"]).clamp(max=self._v_max)
```

硬截断可能掩盖 critic 发散，并使大量 value target 堆积在上界附近。建议优先采用：

- running return normalization；
- Huber loss；
- 对 target 做合理的 percentile clipping；
- 记录超界比例而不是直接截断 value。

如果确实需要限制输出，可以在归一化空间使用 `tanh`，再映射回原始 value 范围。

建议增加以下监控指标：

```text
value_clip_fraction
 target_percentile_01 / target_percentile_99
critic_td_error
advantage_std
V1_minus_V2
```

## 六、使用 multi-step TD 与 GAE 混合 target

长 imagined rollout 使用纯 Monte Carlo return 时方差较大；只使用一步 TD 又可能产生较大偏差。可以混合 multi-step TD 和 GAE：

\[
y=\lambda GAE+(1-\lambda)TD(k)
\]

建议根据 world model 质量调整：

- 真实数据阶段：更偏向 Monte Carlo target；
- imagined rollout 阶段：优先使用 3～5 step TD；
- world model 早期：较短 rollout、较低 \(\lambda\)；
- world model 稳定后：逐步增加 rollout horizon 和 \(\lambda\)。

这样可以让 critic 的 target 随模型质量变化，而不是始终使用固定的 GAE 配置。

## 推荐实施顺序

### 阶段 1：稳定基础 critic

1. 引入 twin critic；
2. advantage 使用 `min(V1, V2)`；
3. 使用 Huber loss；
4. 去掉或替换硬 `v_max` 截断；
5. 增加 value、TD error 和 critic disagreement 日志。

### 阶段 2：加入 world model awareness

1. 计算并维护 prediction error EMA；
2. 如果有 dynamics ensemble，计算 disagreement；
3. 将这些指标拼接到 critic observation；
4. 对 imagined state 使用 conservative value。

### 阶段 3：加入分布和不确定性建模

1. 尝试 8～16 quantile 的 distributional critic；
2. 比较 quantile critic 与 value ensemble；
3. 根据验证环境性能调节不确定性惩罚系数 \(\beta\)。

### 阶段 4：优化 target

1. 引入 3～5 step TD target；
2. 与 GAE 混合；
3. 根据 world model 训练阶段动态调整 horizon 和 \(\lambda\)。

## 实验与消融

至少应比较以下配置：

1. 当前单 critic；
2. twin critic；
3. twin critic + uncertainty features；
4. twin critic + conservative advantage；
5. distributional critic 或 value ensemble；
6. multi-step TD + GAE。

除了最终 tracking reward，还应记录：

- 真实环境成功率；
- imagined rollout 与真实 rollout 的回报差距；
- critic TD error；
- value disagreement；
- OOD 状态比例；
- actor 是否进入高不确定性区域；
- 收敛速度和训练稳定性。

## 建议优先验证的核心方案

最值得优先实现和验证的组合是：

> **Uncertainty-aware twin critic for world-model-based imitation**

其核心流程为：

```text
Twin/value ensemble
    -> value mean + uncertainty
    -> conservative advantage
    -> actor update
    -> reduce exploitation of inaccurate world model
```

这个方向与同时训练 actor 和 world model 的设想兼容。关键约束是：actor 的加速更新必须依赖能够识别 imagined state 不确定性的 critic，否则 actor 可能更快地利用 world model 的错误，而不是真正提升真实环境表现。
