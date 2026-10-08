# Released Artifact Schema

## Terminology

The manuscript uses **summed-concentration absolute error** for the absolute difference between
spatially summed predicted and reference concentration. This quantity is a proxy for integrated
plume concentration. It is not a conservative mass-balance residual.

New metric outputs use `summed_conc_abs_error_*`. Archived evidence may use
`mass_abs_error_*`; figure generation accepts both names without modifying archived values.

## Seed accounting

The current common-test-set comparison uses seeds 0 through 9 for every learned row. Deterministic
baselines are identified separately and do not have a learned-model seed count.

## Statistical tests

Primary manuscript tests are two-sided and seed aligned. The authoritative six-row summary is tables/obj2_acml_primary_metric_two_sided_stats.csv, with inputs in tables/primary_metric_per_seed.csv. The paired standardized difference is d_z. Historical tables/obj2_acml_primary_metric_stats.csv and the headline-significance note are retained unchanged and may contain directional values; they are superseded for manuscript reporting by the explicit two-sided summary. Seed tests quantify training variability under the fixed split, not variability across data splits.

## Provenance

Lightweight CSV and JSON files preserve numerical outputs and seed identifiers. They intentionally
exclude raw fields, checkpoints, full prediction arrays, private filesystem paths, and
collaborator-provided simulation assets.
