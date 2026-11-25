# 连续崎岖地形参考轨迹生成与训练入口

参考 PMT/TCRS 的接触识别、支撑骨盆高度、轨迹修复和 IK 顺序，独立实现固定落点 XY 的完整 LAFAN 动作生成。输入为原始 G1 动作与一个确定的连续崎岖地形。每 256 帧求解一段，32 帧重叠，并对重叠区的修正做加权融合。导出根部、关节位置及与位置修正一致的速度。生成后进行 MuJoCo 足底碰撞检查，对少量剩余穿地帧作平滑的毫米级根部高度修复。地形种子、倍率、来源哈希及输出哈希写入清单。

固定地形种子 1702、倍率 0.25 的 40 个动作已完整生成在 `outputs/g1-support-lafan40-seed1702/`。逐帧审计发现，原始动作有 259809 帧足底穿地超过 5 mm，第一版生成动作仅剩 2 帧，峰值 5.92 mm，均在 `fightAndSports1_subject4`。修复后的该动作重新生成并逐帧审计，穿地超过 5 mm 的帧数为 0，峰值 3.50 mm。第一版整批文件由修复步骤添加前的生成器产生，作为对照留存。训练地形种子应另行生成对应的整批参考。

主线训练入口新增 `--generated-reference-dir`。它检查完整 40 个文件与协议源哈希、帧数、生成文件哈希，按清单设置固定地形种子和高度，并把生成动作提供给原有 motion 命令。此模式使用生成动作的未来参考和跟踪目标，关闭额外的地形高度重置。它尚未经过大规模训练验证。由于现有地形课程动态改变碰撞地形，固定地形预计算文件不能直接用于课程训练。

生成命令为 `.venv/bin/python scripts/prepare_g1_support_reference.py OUTPUT --terrain-seed 1701 --rough-scale 0.25 --replacement-dir outputs/leg-ablation-4k/motions`。逐帧检查命令为 `.venv/bin/python scripts/audit_g1_support_reference.py OUTPUT --replacement-dir outputs/leg-ablation-4k/motions --stride 1`。协议中的 `walk1_subject1` 与当前本地动作文件哈希不同，生成器只接受哈希一致的归档副本。
