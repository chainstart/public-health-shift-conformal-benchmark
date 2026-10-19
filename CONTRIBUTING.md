# Extending the benchmark

Start with [the method guide](docs/METHODS.md) and the synthetic quick start.
A method contribution should provide a uniquely named `MethodSpec`, deterministic
use of the supplied seed, documented information access and settings, and a small
contract test. Evaluate it with the same prediction manifest, budgets, alpha and
shared draws as the baselines. Keep scientific results under a named configuration;
do not overwrite reference results or choose settings from test outcomes.

For a dataset/predictor contribution provide public acquisition/version recipes,
preprocessing and exclusions, stable identifiers, deterministic role assignment,
role-overlap audits, source-only training code and hash-identified predictions.
State how the new task differs from the standard twelve-task configuration. See
[the benchmark specification](docs/BENCHMARK.md) and the existing preparation
code for the reference convention. Raw records and model arrays remain local.

Run `python -m unittest discover -s tests -v` before proposing a change.
Full raw-data reconstruction and optimizer experiments are separate, longer
checks; report exactly which checks and numerical tolerances were used.
