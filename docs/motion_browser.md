# MuJoCo motion browser

Run the interactive Viser browser with the official checkpoint:

```bash
bash scripts/play_motion_browser.sh
```

The browser provides a `Motion Clip` dropdown. Selecting another clip resets
the robot and reference ghost to a random frame from that clip. The solid robot
is the policy rollout; the green translucent robot is the reference motion.
The standard Viser controls also provide pause, single-step, reset, camera
tracking, and playback speed controls.

To use another checkpoint or subset of motions:

```bash
MODEL_DIR=logs/SuperMimic-GenCoord-G1/<run> \
MOTION_GLOB='data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk*.npz' \
bash scripts/play_motion_browser.sh
```
