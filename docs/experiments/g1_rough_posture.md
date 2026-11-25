# 连续崎岖地形的支撑足姿态与骨盆参考实验

使用 oracle 第 5000 轮 Actor，沿用 9 个 LAFAN 片段、地形种子 1702、崎岖度倍率 0.25、共同起点和前 6 步共同预热。A 为固定落点净空 IK，B 在 A 的基础上用支撑足底点拟合局部平面，并在 8° 内调整脚掌俯仰与横滚。C 在 B 的基础上将骨盆高度向支撑脚高度变化移动，同时将骨盆水平位置朝支撑中心移动，每轴最多 1.5 cm。所有组都保持原落点 XY 与原接触时序。三组 Actor 和起点文件的 SHA-256 一致。

| 参考 | 可配对触地片段 | 骨盆竖直速度峰值 m/s | 躯干倾角峰值 ° | 支撑脚滑速 m/s | 提前终止 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A 净空 IK | 5 | 0.549 | 33.00 | 0.272 | 0/9 |
| B 足姿态适配 | 5 | 0.520 | 32.60 | 0.274 | 1/9 |
| C 足姿态与骨盆适配 | 5 | 0.531 | 32.71 | 0.268 | 1/9 |

骨盆速度和躯干倾角是同侧触地时间相差不超过 100 ms 的 5 个片段中，实际触地后 200 ms 峰值的片段均值。脚滑按全部有效支撑时间加权。B/C 的 18 段支撑足姿态修正平均 6.13°，最大达到 8° 限幅。sprint1_subject4 在 A 中未提前终止，在 B/C 中分别于第 30/27 步提前终止。当前结果提示局部姿态适配可能改善部分落地冲击，但稳定性未获改善。由于配对样本少，且先前固定落点回放的终止结果不同，不能据此宣布收益。

运行 `.venv/bin/python scripts/eval_g1_rough_posture.py outputs/g1-ankle-closed-loop/checkpoint outputs/g1-ankle-optimization-probe outputs/leg-ablation-4k/protocol.json outputs/g1-ankle-closed-loop/starts.npz outputs/g1-rough-posture`。原始指标、每个片段的参考姿态及接触轨迹位于 `outputs/g1-rough-posture/`。
