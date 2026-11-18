# 物理先验消融实验阶段性对比

当前结果均来自相同的 1K 训练预算、seed=42 和 sim-calibration epochs=2 配置。奖励为训练日志中的阶段末平均奖励，速度为日志中的 steps/s。

| 版本 | 改动 | 1K奖励 | 速度 |
|---|---|---:|---:|
| Official SuperMimic | 官方实现 | 10.18 | 约16.4k |
| B | 增量状态预测 | 12.72 | 约13.9k |
| L | 固定 progress LR | 12.60 | 约15.0k |
| R | World Model residual correction | 3.29 | 约10.0k |
| P | reference residual policy | 1.69 | 约15.1k |
| H | mirror augmentation | 约1.42（约570轮） | 约12.1k |
| G | CUDA graph collection | 待完成 | 待完成 |

## 阶段判断

B 在早期奖励上高于官方版本。L 与 B 接近，说明 progress LR schedule 不是主要退化来源。R 和 P 的早期奖励明显下降，优先怀疑残差分支的初始化、损失权重或梯度尺度。H 尚未完成 1K，不能据中途结果下结论。G 尚未启动。

## 连接状态

2026-11-18 继续监控时，服务器 SSH 密码认证失败，因而 H 的最终日志和 G 的启动状态尚未重新核验。恢复服务器连接后，应先读取 H 日志末尾，再启动 G，最后用相同字段补齐表格。
