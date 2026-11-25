# 支撑期骨盆高度重建的首轮验证

参考 Mondo Robotics PMT/TCRS 提供的接触阶段、支撑脚与骨盆高度联动处理顺序，在本仓库独立实现了一个离线原型。参考代码提交为 `1a92390077d329d22cbcdbf89c4a5c982003fbc8`。本次使用连续崎岖地形，保留原落点 XY 和足姿态。先从名义足底高度与踝部速度识别支撑期，按支撑脚地形高度变化得到骨盆高度目标。目标经过平滑与每帧限速后用于有界 IK；腾空期间回到原始骨盆轨迹。相机深度与大规模训练尚未接入。

9 个原有失败窗口，共 124 个检测到的支撑样本。相同 oracle 5K Actor、相同起点、地形种子 1702、崎岖度倍率 0.25。两次配对回放结果如下。

| 回放 | 参考 | 可配对触地片段 | 骨盆竖直速度峰值 m/s | 支撑脚滑速 m/s | 提前终止 |
| --- | --- | ---: | ---: | ---: | ---: |
| 第一轮 | 现有净空 IK | 7 | 0.554 | 0.292 | 1/9 |
| 第一轮 | 支撑期骨盆重建 | 7 | 0.507 | 0.269 | 0/9 |
| 第二轮 | 现有净空 IK | 5 | 0.527 | 0.271 | 1/9 |
| 第二轮 | 支撑期骨盆重建 | 5 | 0.528 | 0.311 | 1/9 |

两组参考的穿地超过 5 mm 帧数均为 0。现有参考的最大穿地约 1.9 mm，新参考约 2.8 mm。骨盆参考速度 95 分位与最大值分别从 0.648、0.802 m/s 降至 0.622、0.738 m/s。第一轮闭环有所改善，第二轮没有复现，现阶段只能证明该生成流程可运行且不会造成明显参考穿地。短窗口的触地配对片段数随回放变化，暂不足以支持训练主线替换。

复现命令为 `.venv/bin/python scripts/eval_g1_support_root.py outputs/g1-ankle-closed-loop/checkpoint outputs/g1-ankle-optimization-probe outputs/leg-ablation-4k/protocol.json outputs/g1-ankle-closed-loop/starts.npz outputs/g1-support-root`。原始结果分别在 `outputs/g1-support-root/` 与 `outputs/g1-support-root-repeat/`。
