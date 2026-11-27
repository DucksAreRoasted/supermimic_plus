# raw 里程碑与在线参考调研 · 2026-11-27

raw 对照提交 10f38d564a502ee65f0f575c830305792ead3e3d，复现入口 scripts/eval_g1_rolling_reference.py 的 raw 组。冻结权重：outputs/g1-support-30k-seed1701/checkpoint-latent-30k/actor.pt。四个种子的原始起点及 raw 轨迹保存在outputs/g1-reference-10f38d5/s*/ 下。文件校验见 raw-milestone.sha256。

以下为 20 个不同的研究或开源项目。右侧为对本实验的启发，不表示复现过各项目或取得相同性能。

1. [Perceptive BFM](https://acodedog.github.io/perceptive-bfm/) — 借鉴接触感知的摆动脚净空和足底几何；把适配放在运行时是本实验自己的设计。
2. [NEXUS](https://nexus-humanoid.github.io/) — 原始动作与机载感知作为输入；采用相同原始指令比较。
3. [PGMT](https://luyili.github.io/pgmt/) — 局部放宽受地形影响的跟踪目标；保留整体意图。
4. [TAGA](https://marmotlab.github.io/taga-humanoid/) — 感知覆盖与运动方向相关；分前进和后撤报告可见率。
5. [BeyondMimic](https://beyondmimic.github.io/) — 跟踪器作为独立低层模块；冻结后检验上游输入变化。
6. [SuperMimic](https://github.com/HybridRobotics/supermimic) — 保留原始 G1 跟踪入口及权重。
7. [Elevation Mapping](https://github.com/leggedrobotics/elevation_mapping) — 随机器人运动融合局部高度及不确定性。
8. [Elevation Mapping CuPy](https://github.com/leggedrobotics/elevation_mapping_cupy) — 参考批量更新局部地图的实现；记录地图计算延时。
9. [Grid Map](https://github.com/ANYbotics/grid_map) — 分别保存高度、置信度与年龄等地图层。
10. [Traversability Estimation](https://github.com/leggedrobotics/traversability_estimation) — 将落点坡度、粗糙程度与可见性分别评价。
11. [Invariant EKF](https://github.com/RossHartley/invariant-ekf) — 区分实测本体及接触辅助状态估计与仿真位姿真值。
12. [LIPM Walking Controller](https://github.com/stephane-caron/lipm_walking_controller) — 落脚改变后检查身体支撑与重心过渡。
13. [TOWR](https://github.com/ethz-adrl/towr) — 足端和身体约束需统一；第一版锁定参考时序，避免混淆。
14. [OCS2](https://github.com/leggedrobotics/ocs2) — 区分支撑和摆动阶段的约束。
15. [Bipedal Locomotion Framework](https://github.com/gbionics/bipedal-locomotion-framework) — 区分足步计划、摆动脚轨迹和低层跟踪。
16. [TSID](https://github.com/stack-of-tasks/tsid) — 按任务分配接触、身体姿态和足端约束。
17. [Pink](https://github.com/pink-kinematics/pink) — 用加权任务及显式关节限制拟合参考。
18. [Mink](https://github.com/kevinzakka/mink) — MuJoCo 差分 IK 接口可作为数值校验。
19. [PyTorch Kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) — 批量 FK/Jacobian 可用于核对在线两路参考是否一致。
20. [MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc) — 滚动更新只依赖当前状态；比较规划周期与控制周期。

## 实现与配对协议

两组使用相同的冻结 Actor、原始动作、保存的模拟器起点和在线相机。raw 按原相位读取 LAFAN 动作。fixed_phase 每次用新深度与近期地图查询名义足底，按接近支撑面的程度估计骨盆的公共高度修正，再将左右脚的相对高差通过阻尼腿部雅可比转换为关节修正。相机点云按机器人连杆位置近似剔除自身。修正只改变骨盆高度和腿部关节，由同一段姿态重建运动编码目标、参考速度与双脚未来姿态。规划器全程未得到观测时保留原指令。模拟器提供机器人位姿，位姿估计误差尚未验证。

本地 raw 里程碑压缩包 SHA256 为 `3b7ef0aeb9ece65f8831fbdde4c2b736d45024055a33ad34935d25891894513a`。Actor SHA256 为 `5826dc4cffad4adbf5837529f5f8b1b865c9f8474282bb0dda742dfbbb4fb1f9`。权重与 LAFAN 数据需要单独准备。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_g1_rolling_reference.py \
  --checkpoint outputs/g1-support-30k-seed1701/checkpoint-latent-30k \
  --motion data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk2_subject1.npz \
  --fixed-phase-pair --steps 180 --terrain-seed 1715 --seed 42 \
  --scenes forward return backward slow_backward --output outputs/fixed-phase-reproduction
```

复用此命令可更换地形与模拟种子、指定新的输出目录。配对评测先比较末相位、提前终止，再按同一时刻的原始骨盆水平位置检查实际进度。第 25、50、…、175 帧及末帧共 8 个检查点采用 15 cm 容差。触地脚底水平速度取接触样本均值。每段观测 180 步；这只覆盖所截片段，不代表完整动作完成率。

## 2026-11-27 配对结果

代码 df6d764c62fecedcc2b6db33b2c2401f91863b05。四个开发地形配对 (1715,42)、(1716,43)、(1721,44)、(1730,46) 使用提交前工作树，第五个地形 (1781,47) 使用干净提交复核。每种地形使用同一个行走动作的四个片段。以下为 20 次回放等权平均或计数，检查点共 160 个。

| 指标 | raw | fixed_phase |
|---|---:|---:|
| 平均末相位，初始为 0 | 179 | 179 |
| 限时内提前终止 | 0/20 | 0/20 |
| 末帧到达原始水平目标 | 20/20 | 20/20 |
| 同时刻水平检查点命中 | 139/160 | 135/160 |
| 骨盆实际起末水平位移模长 | 1.520 m | 1.515 m |
| 骨盆到同时刻原始水平参考的平均距离 | 9.19 cm | 9.24 cm |
| 触地脚底平均水平速度 | 15.97 cm/s | 16.12 cm/s |

按前进、折返、后退、慢后退排列，检查点命中依次为 raw 32、29、38、40，fixed_phase 29、29、37、40（每项 40）。脚底接触速度依次为 raw 14.39、18.76、19.24、11.50 cm/s，fixed_phase 14.90、20.61、18.22、10.73 cm/s。常规回放中最大骨盆高度修正 4.5 cm，最大腿关节修正 0.18 rad。补充对照将规划器深度遮挡 180 步，当前与提前 4 步的两脚参考逐帧均与 raw 相同，实际骨盆坐标最大差 3.08 mm。该对照只遮挡规划器，Actor 保留原始传感输入。raw 继续作为性能基线。
