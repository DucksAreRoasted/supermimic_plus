# SuperMimic

A [SuperTrack](https://montreal.ubisoft.com/en/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning/)
implementation for training world models and policies to mimic motion data, built on [mjlab](../mjlab_research).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:kevinzakka/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## LAFAN multi-motion critic comparison

One shared policy and world model train across all matched clips. With the
prepared LAFAN dataset, the default `npz/*.npz` selects all 40 motions:

```bash
bash scripts/run_g1_critic_comparison.sh
```

Both runs use the official training defaults: 4096 environments, replay ratio
16, WM `num_batches=10, batch_size=2048, horizon=8`, and policy
`num_batches=1, batch_size=4096, horizon=32`. The default iteration limit is
1,000,000. Both variants use the same seed (`SEED=42` by default).
`MAX_ITERATIONS=40000` explicitly overrides that budget; `CLIP=walk1_subject1`
selects just one clip for a small experiment.

`--runner.critic-variant official` uses the unchanged upstream single critic
from commit `5efbf114e11f725c31d83e217be987c13c0e4ce2` (MSE and value/return
clamps). `--runner.critic-variant twin` selects the separate experimental twin
learner. The former `--runner.policy.twin-critic` switch is replaced by this
explicit choice. Earlier 40k single-critic checkpoints used the modified
learner and are not official baseline results.

## Local 40k playback

Open the downloaded G1 checkpoint in the native MuJoCo viewer:

```bash
bash scripts/play_g1_40k.sh twin walk1_subject1
bash scripts/play_g1_40k.sh single walk1_subject1
```

The solid robot follows the policy; the green translucent robot shows the
reference motion at the same frame. Space pauses, Enter resets, and R toggles
the reference visualization. The launcher uses CUDA for physics and inference
and Mesa software rendering for compatibility with this desktop's GLX setup.
This can run slower than real time. `LIBGL_ALWAYS_SOFTWARE=0` and
`__GLX_VENDOR_LIBRARY_NAME=nvidia` can override the rendering defaults on a
compatible display.

Other local checkpoints using the standard runner architecture can be loaded
with `play.py TASK --model-dir PATH --motion-file FILE --viewer native`.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
