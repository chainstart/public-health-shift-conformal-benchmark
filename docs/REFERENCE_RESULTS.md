# Reference results and paper case study

The reusable benchmark defines evaluation rules and accepts new methods. The
paper case study is one completed application, including both its historical
primary analysis and separately labelled additional experiments.

| Reference | Location | How to use |
| --- | --- | --- |
| Maintained interface fixed-300 run | `reference/fixed300/` | Four baselines plus an external-method integration example; exact settings and receipts included |
| Historical primary results | `examples/legacy-experiments/results/public_benchmark_*.csv` | Audit original paper protocol; do not compare as if outcome-blind |
| Primary reconstruction and original code | `examples/health-study/primary-protocol/` | Reconstruct historical primary aggregates |
| Additional basic-method budgets | `examples/health-study/results/` | Follow the named protocol and `REPRODUCE.md` |
| Fixed-300 nine-method comparison | `examples/health-study/results/matched/` | Stronger comparator reference; preserve labels and optimizer settings |
| Cardiac and sepsis extensions | `examples/health-study/institutional/` | Separate external case studies |
| Full paper presentation | `examples/health-study/supplement/` | Published supplementary discussion, tables and figure sources |

`examples/health-study/` is copied from published support v1.4.15,
[10.5281/zenodo.23461041](https://doi.org/10.5281/zenodo.23461041).
It includes code and small derived reference summaries, not raw patient records,
trained weights or prediction arrays. The source archive's inventory and
reproduction instructions retain their original meaning within this directory.

The interface smoke fixture is synthetic and belongs to none of these scientific
result sets. A real-data evaluation must identify the actual prediction manifest,
configuration and method inputs. Do not replace historical reference files when
running a new experiment; write outputs under `work/`.
