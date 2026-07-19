# Legacy Results

The score files in this directory were produced before the 2026-06-13 BTWM
correctness fixes. They must not be used for paper comparisons or statistical
claims.

Known issues in those runs:

- inverse-action targets were shifted to the wrong transition;
- discrete heads used one extra class because action-space `high` is exclusive;
- continuous actions were not converted to classifier bins;
- runs used inconsistent Atari settings and often exceeded 100,000 steps;
- baseline and BTWM runs were not consistently seed- or compute-matched.

Valid replacement runs use directories named:

```text
runs/<model>_<domain>_<task>_seed<seed>/
```

and are launched through `code/run_experiment.sh`.
