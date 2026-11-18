# Reference residual 性能下降定位实验

## 已知现象

服务器上的 `g1-reference-residual-beta1-replay16-epochs2-10000it-seed42` 最终奖励约为 20.48，明显低于 B 版本约 39。该运行同时启用了多个改动，无法归因。

单独的 `g1-residual-correction-simcal-epochs2-10000it-seed42` 最终奖励约为 39.96，但速度约 9.2k FPS，说明 World Model residual head 本身没有造成奖励崩溃，主要代价是计算速度。

## 固定条件

所有实验统一使用以下配置：

- G1、40 个 LAFAN motion 文件、seed=42；
- 4096 environments、replay ratio=16；
- sim-calibrated Critic、epochs=2、horizon=32、weight=0.1；
- B 的增量状态预测和半隐式积分；
- 10K iterations，保存 1K checkpoint；
- 记录 Mean reward、Policy reward、episode length、total FPS、World Model loss、Critic loss、梯度范数和显存。

每个实验只改变一个开关，run name 必须包含所有非默认开关。

## 第一阶段短跑筛选

每个实验先跑 1K iterations。保留与 B 的平均奖励差距小于 20%，且没有 NaN、梯度爆炸或 FPS 下降超过 20% 的配置，再进入 10K 复验。

| 编号 | 在 B 上新增的改动 | 目的 |
|---|---|---|
| A | B，不加任何新增组件 | 当前对照 |
| R | 只开 World Model residual correction | 确认 residual head 的奖励和速度影响 |
| P | 只开 reference residual policy wrapper | 判断策略 residual 是否是主因 |
| Z | P + residual policy zero initialization | 判断初始化是否导致早期学习停滞 |
| M | P + MuonAdam | 判断优化器是否破坏策略更新 |
| L | P + progress learning rate | 判断学习率调度是否导致后期退化 |
| H | P + mirror augmentation | 判断镜像数据是否造成标签或动作映射问题 |
| G | P + CUDA graph collection | 判断采集图是否改变训练语义 |

短跑阶段不要组合开关。若 P 已经明显失败，先停止 P 后面的组合实验，继续验证 R、M、L、H、G 的独立影响。

## 第二阶段归因实验

对第一阶段表现正常的组件分别做 10K 训练。之后只构造两个组合：

1. `B + R + E`，验证动力学 residual 与 simulator-calibrated Critic 的目标组合。
2. `B + R + E +` 一个通过第一阶段筛选的策略组件。

每次组合只增加一个已经通过短跑的组件。组合失败时回退到上一个成功配置。

## 重点诊断

### 策略 residual

比较 `P` 和 `Z` 的 Actor loss、Policy reward、动作均值和动作方差。如果 zero initialization 能恢复奖励，问题在 residual policy 初始输出或 residual 分支的梯度比例。

### MuonAdam

比较参数更新范数、梯度范数和 entropy。若 reward 在前 1K 轮正常而随后突然下降，检查 optimizer step、weight decay 和参数组是否覆盖了原始 Actor。

### Mirror augmentation

单独检查镜像前后的 state、action、goal 和 reward。若 `H` 在第一轮就明显低于 A，优先检查关节名称配对、左右轴符号、四元数变换和动作缩放。

### Progress learning rate

记录每个 iteration 的实际 learning rate。关闭调度后复跑 P。若关闭后恢复，保留固定学习率或重新限制调度范围。

### CUDA graph collection

比较 G 与 P 的 reward 和每步 collection time。若速度提升但 reward 变化，检查 graph capture 是否复用了旧的 reset、reward 或 terminated buffer。

## 判定规则

- 奖励下降超过 20% 且只发生在一个单变量实验中，该变量列为高危改动。
- 奖励接近 B 但 FPS 下降超过 20%，该变量属于性能开销问题，可以单独优化。
- 只有组合实验失败时，检查组件之间的交互，不能把失败归因给任一单变量。
- 最终候选必须在 10K 的 8K 和 10K checkpoint 都不低于 B 的奖励，同时记录速度和显存代价。

## 执行顺序

1. A、R、P、Z、M、L、H、G 各跑 1K。
2. 根据短跑结果选出高危变量和可保留变量。
3. 对 R 和通过筛选的单变量跑 10K。
4. 加入 E，跑 `B + R + E` 10K。
5. 只把通过验证的单个策略组件加入最终组合。

