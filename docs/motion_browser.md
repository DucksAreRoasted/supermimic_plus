# MuJoCo motion browser

Run the interactive local MuJoCo window with the official checkpoint:

```bash
bash scripts/play_motion_browser.sh
```

The native window shows the policy rollout and green translucent reference
ghost. Native viewer controls provide pause, single-step, reset, camera
tracking, and playback speed controls. Motion clip selection is available in
the native MuJoCo window with `[` / `]` (previous/next clip); each switch
resets to a random frame in the selected clip.

To use another checkpoint or subset of motions:

```bash
MODEL_DIR=logs/SuperMimic-GenCoord-G1/<run> \
MOTION_GLOB='data/modelscope/LAFAN1_Retargeting_Dataset/npz/walk*.npz' \
bash scripts/play_motion_browser.sh
```

To use the browser UI instead:

```bash
VIEWER=viser bash scripts/play_motion_browser.sh
```
