# 固定骨盆的双腿 IK 配对回放

使用同一个 oracle 5K Actor、9 个片段、地形种子 1702、崎岖度倍率 0.25 和相同保存起点。两组使用同一足底净空目标、固定落点 XY、原足姿态和相同的有界 IK。现有方案允许骨盆高度参与 IK，纯腿方案固定整个骨盆姿态和位置，仅优化双腿 12 个关节。每组从同一原始动作执行前 6 步，再切换参考。

| 回放 | 方案 | 配对触地片段 | 骨盆竖直速度峰值 m/s | 躯干倾角峰值 ° | 支撑脚滑速 m/s | 提前终止 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 第一次 | 骨盆高度 + 双腿 IK | 6 | 0.506 | 30.75 | 0.247 | 0/9 |
| 第一次 | 纯双腿 IK | 6 | 0.536 | 31.38 | 0.264 | 1/9 |
| 第二次 | 骨盆高度 + 双腿 IK | 8 | 0.545 | 30.88 | 0.249 | 1/9 |
| 第二次 | 纯双腿 IK | 8 | 0.524 | 30.72 | 0.250 | 0/9 |

两次回放的优劣排序翻转。同一组参考的实际接触时间也有变化，因此“配对触地片段”的集合不同。不能从这 9 个短窗口断言纯双腿 IK 更好或更差。纯腿方案的参考骨盆位置逐帧与原始动作完全相同，现有方案相对原始动作的最大骨盆高度变化约 5.4 mm。回放没有重新训练 Actor。

运行 `.venv/bin/python scripts/eval_g1_leg_only.py outputs/g1-ankle-closed-loop/checkpoint outputs/g1-ankle-optimization-probe outputs/leg-ablation-4k/protocol.json outputs/g1-ankle-closed-loop/starts.npz outputs/g1-leg-only`。两次回放的逐片段指标与参考姿态分别位于 `outputs/g1-leg-only/` 和 `outputs/g1-leg-only-repeat/`。
