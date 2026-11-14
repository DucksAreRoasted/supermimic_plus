# Experiment naming

Every training run must provide a descriptive `--runner.run-name`. Use this
shape whenever possible:

`<robot>-<critic>-<aggregation>-<dataset>-<iterations>it-seed<seed>`

Examples:

- `g1-official-single-critic-lafan40-40kit-seed42`
- `g1-twin-mean-lafan40-40kit-seed42`
- `g1-twin-min-from-mean-5kit-seed42`
- `g1-sim-calibrated-lafan40-40kit-seed42`

The runner stores TensorBoard events, checkpoints, and `run_manifest.json` in
`logs/<task>/<run-name>/<timestamp>/`. The manifest records the full command,
critic variant, aggregation, iteration budget, device, warm-start directory,
and other resolved run metadata. Shell runners also write their console output
to a matching `<run-name>.log` file under a labelled experiment log folder.

Do not use timestamp-only names or generic names such as `run`, `test`, or
`latest`; those make copied experiments impossible to identify reliably.
