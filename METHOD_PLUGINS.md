# Method interfaces

For new methods use the maintained [method development guide](docs/METHODS.md)
and `python -m benchmark evaluate`. The [quick start](README.md#quick-start-your-method-against-four-baselines)
runs a user method and four baselines through the same evaluation pipeline.

The earlier `examples/legacy-experiments/scripts/calibration_plugins.py` CSV interface remains available for
compatibility. It is separate from the original full paper runner and does not
provide the current unified evaluation/aggregation workflow.
