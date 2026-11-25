# 连续崎岖地形参考轨迹生成与训练入口

参考 PMT/TCRS 的接触识别、支撑骨盆高度、轨迹修复和 IK 顺序，独立实现固定落点 XY 的完整 LAFAN 动作生成。输入为原始 G1 动作与一个确定的连续崎岖地形。每 256 帧求解一段，32 帧重叠，并对重叠区的修正做加权融合。导出根部、关节位置及与位置修正一致的速度。生成后进行 MuJoCo 足底碰撞检查，对少量剩余穿地帧作平滑的毫米级根部高度修复。地形种子、倍率、来源哈希及输出哈希写入清单。

固定地形种子 1702、倍率 0.25 的 40 个动作已完整生成在 `outputs/g1-support-lafan40-seed1702/`。逐帧审计发现，原始动作有 259809 帧足底穿地超过 5 mm，第一版生成动作仅剩 2 帧，峰值 5.92 mm，均在 `fightAndSports1_subject4`。修复后的该动作重新生成并逐帧审计，穿地超过 5 mm 的帧数为 0，峰值 3.50 mm。第一版整批文件由修复步骤添加前的生成器产生，作为对照留存。训练地形种子应另行生成对应的整批参考。

主线训练入口新增 `--generated-reference-dir`。它检查完整 40 个文件与协议源哈希、帧数、生成文件哈希，按清单设置固定地形种子和高度，并把生成动作提供给原有 motion 命令。此模式使用生成动作的未来参考和跟踪目标，关闭额外的地形高度重置。它尚未经过大规模训练验证。由于现有地形课程动态改变碰撞地形，固定地形预计算文件不能直接用于课程训练。

生成命令为 `.venv/bin/python scripts/prepare_g1_support_reference.py OUTPUT --terrain-seed 1701 --rough-scale 0.25 --replacement-dir outputs/leg-ablation-4k/motions`。逐帧检查命令为 `.venv/bin/python scripts/audit_g1_support_reference.py OUTPUT --replacement-dir outputs/leg-ablation-4k/motions --stride 1`。协议中的 `walk1_subject1` 与当前本地动作文件哈希不同，生成器只接受哈希一致的归档副本。

## 训练地形种子 1701 的数据与试验

40 个动作已在训练地形种子 1701、倍率 0.25 上重新生成。对完整帧数做 MuJoCo 足底碰撞审计：原始参考有 242732 帧穿地超过 5 mm；适配后为 0 帧，最大剩余穿入 3.96 mm。本地和服务器分别生成并得到相同的审计总数。所有文件的来源、帧数和输出哈希均由清单核验。

服务器独立工作目录 `/root/gpufree-data/g1-support-main-code` 使用提交 `97f9cc03f6b2552b80959d22e3bff5d329cc07b2`，原实验工作目录未改动。单轮冒烟训练已完成并保存 checkpoint，确认 40 个动作、单帧本体输入、68 维参考目标和固定地形种子均被加载。

从零训练配置：`--generated-reference-dir /root/gpufree-data/g1-support-lafan40-seed1701 --iterations 10000 --num-envs 2048 --batch 2048 --policy-batch 2048 --actor-input state1 --goal-width 68 --camera-profile instinct --cache-reference --official-backends --training-reference /root/gpufree-data/g1-rough-reference-5k/training-reference.json --seed 42`。日志为 `/root/gpufree-data/g1-support-train-10k.log`。这里的地形适配目标是由仿真真值预计算得到的特权参考，训练结果只检验其闭环跟踪效果；尚不能证明深度图本身能合成参考。

闭环评估使用 `scripts/eval_scene_flat_regression.py CHECKPOINT docs/experiments/lafan40_seed42.json STARTS OUTPUT --generated-reference-dir /root/gpufree-data/g1-support-lafan40-seed1701 --save-starts`。评估器从生成清单核对 40 个文件的来源、长度和输出哈希，使用清单里的固定地形，直接回放已适配的参考，避免再次进行骨盆高度重置。汇总包含动作完成数、完成帧数、全身位置与关节误差、脚底支撑期滑动速度，以及每个动作的提前终止原因。正式结果和复跑差异记录如下。

## 10K 闭环训练与参考消融

固定地形参考的从零训练完成 10K 轮，2048 个环境，约 26 分钟；最终 checkpoint 在服务器训练目录的 `checkpoints/iter-010000`。第 5K 和 10K 轮用本地同一组起点回放 40 个动作，首次结果的完成数为 15 和 20，平均完成帧比例为 63.9% 和 71.8%，平均身体位置误差为 8.3 cm 和 5.8 cm，平均支撑期脚底速度为 0.183 和 0.155 m/s。最终训练奖励约 31–33，不能代替物理回放指标。

评估的**运行间波动很大**：同一 10K Actor、同一适配参考、相同起点和校验值，40 环境 GPU 回放重跑的完成数从 20 变为 30。MuJoCo Warp 的随机种子接口本身注明物理仿真未完全确定。因而一次原始参考的 28/40、一次“18 个移动动作适配、22 个其他动作保留原版”的 24/40，不能证明某种参考稳定优于另一种。该混合版由 `scripts/compose_g1_support_reference.py` 构造，只用于消融；不作为训练定档。

为降低长时程波动，在 `fightAndSports1_subject4`、`jumps1_subject1`、`jumps1_subject2`、`fallAndGetUp1_subject4` 的指定参考帧 900、1200、500、1100 上，分别用单环境 CPU 回放 300 步。相同配置的重复回放逐值一致，原始与完整适配都存活 4/4。按四段等权，原始参考的身体位置误差为 8.21 cm、脚底支撑期速度为 0.219 m/s；完整适配分别为 8.87 cm 和 0.178 m/s。适配参考消除了这四个完整动作超过 5 mm 的静态脚部穿地，并降低脚滑，但有约 0.66 cm 跟踪误差代价。这些是选中的开发窗口，不能外推为 40 动作总体因果效应。

两个修正消融都未胜出：非移动动作保留 50% 几何修正时，40 个动作仍有 61192 帧脚部穿地超过 5 mm；`--swing-clearance-only` 仅在非支撑脚将穿地时修正，四个动作都保持 0 帧超过 5 mm 穿地，CPU 短窗口平均身体位置误差为 8.75 cm，但脚底速度回升至 0.222 m/s。故当前仍使用完整适配生成器；接下来的实验应优先验证视野递推与长期策略效果，并在新的参考算法上同时设立几何、跟踪和脚滑门槛。
