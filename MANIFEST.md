# Interface release inventory

Version 2.0.0, 2026-10-19. The maintained benchmark is `benchmark/`, its user
instructions are in `README.md` and `docs/`, and its method example is
`examples/source_split.py`. The v1.4.15 experimental-code case study with manuscript text excluded
is `examples/health-study/`; see its own `CONTENTS.md` and `UPSTREAM.json`.

The complete previous GitHub tree, including original scripts, result CSVs,
figures and tables, is preserved under `examples/legacy-experiments/`. Fixed URLs
to the historical tag and commits remain valid. The [historical manifest](docs/HISTORICAL_MANIFEST.md) describes
that July package, not the coverage of the new method-development interface.
`examples/legacy-experiments/scripts/validate_release.py` remains a lightweight historical-file validator.
`python -m unittest discover -s tests -v` exercises the maintained interface.
See [verification](docs/VERIFICATION.md) for the actual tested scope.

`CHECKSUMS.sha256` records all distributed files except itself. Raw records,
probability arrays, generated workspaces, virtual environments and caches are
excluded from the public inventory.
