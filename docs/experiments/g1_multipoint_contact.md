# 支撑脚多点 IK 小规模消融

参考 [Perceptive BFM 的 TCRS](https://github.com/Mondo-Robotics/PMT/blob/main/TCRS/README.md) 中支撑脚的脚踝、脚尖、脚跟多点约束，在现有有界窗口 IK 上增加可切换的物理脚底前后端点目标。端点由当前 G1 MuJoCo 碰撞几何计算，不使用论文模型的固定偏移。只有原平地动作中处于支撑期、且该端点本来低于 12 mm 的样本才启用约束，保留脚跟离地与脚尖离地动作。原骨盆高度、踝部目标、关节限位、时间平滑与 Actor 均不变。

使用原有 9 个失败窗口、oracle 5K Actor、地形种子 1702、粗糙度 0.25、同一个保存的起点。对照为现有支撑骨盆高度 + 踝部 IK；实验组只额外添加前后脚底点的地形目标。第一版将原脚底点高度直接加地形高度，`dance2_subject3` 有 13 帧足底穿地超过 5 mm，因此未进入闭环。给接触目标留出 4 mm 最小净空后，9 个窗口两组都没有超过 5 mm 的足底穿地，最大穿入均约 2.81 mm。实验组启用了 120 个端点样本；与对照相比，腿关节参考最大变化 0.032 rad，骨盆高度最大变化 0.81 mm。

在 9 环境 MuJoCo Warp 闭环上重复两次。第一次对照与多点组分别有 1/9、0/9 提前终止，支撑期脚底速度分别为 0.267、0.268 m/s；第二次分别有 1/9、1/9 提前终止，脚底速度为 0.275、0.283 m/s。配对触地片段数量分别为 6 和 7，两个版本的触地后骨盆竖直速度与躯干倾角接近。现有证据只表明多点 IK 可运行且未破坏静态几何，**没有证明闭环收益**；本轮不据此替换整批参考或启动大规模训练。

复现命令：

```bash
PYTHONPATH=$PWD/src:$PWD/scripts .venv/bin/python scripts/eval_g1_multipoint_contact.py \
  outputs/g1-ankle-closed-loop/checkpoint \
  outputs/g1-ankle-optimization-probe \
  outputs/leg-ablation-4k/protocol.json \
  outputs/g1-ankle-closed-loop/starts.npz \
  outputs/g1-multipoint-contact
```

两次回放原始结果分别在 `outputs/g1-multipoint-contact/` 和 `outputs/g1-multipoint-contact-repeat/`。这 9 个旧失败窗口不是完整动作分布；MuJoCo Warp 闭环结果存在运行间波动，1 个提前终止的差异不足以定档。
