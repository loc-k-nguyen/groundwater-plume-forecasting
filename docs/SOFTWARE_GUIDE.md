# Software guide

## Navigate the implementation

| Component | Entry point | Role |
|---|---|---|
| Forecasting | `src/models/ta_anp.py` | Attention, latent prior and residual decoding |
| Sequence inputs | `src/data/dataset_interp.py` | Context/query indices and deterministic spatial patches |
| Training | `scripts/train_ta_anp.py` | Recorded TA-ANP settings; requires authorized fields |
| Prediction export | `scripts/export_prediction_cache.py` | Private checkpoint evaluation and cache export |
| Plume diagnostics | `scripts/compute_plume_metrics.py` | Similarity, displacement and summed-concentration diagnostics |
| Statistical comparisons | `scripts/paired_significance_tests.py` | Seed-aligned paired tests with input validation |
| Evidence verification | `scripts/verify_revision_evidence.py` | Recompute six archived primary comparisons |
| Integrity | `scripts/check_release.py`, `scripts/build_manifest.py` | Release scope, source hashes and file-set verification |
| Presentation | `scripts/make_figures.py`, `scripts/make_research_overview.py` | Aggregate plots without simulation fields |

The model and training components are research implementations, not a production forecasting service. The included tests use small synthetic inputs and released aggregate evidence. A passing test suite does not establish field-data accuracy, calibrated uncertainty or independent-geology transfer.

## Reproduction levels

Public files support arithmetic and statistical verification of the released seed summaries, small CPU model checks and aggregate plotting. Recomputing cache-level metrics requires excluded prediction caches. Full inference requires authorized fields and trusted checkpoints; full retraining additionally requires the archived environment and data access. Complete trainers for every comparison family are not included.

Keep the earlier-target reference protocol distinct from supervised forecasting. Keep software versions distinct from manuscript versions and record the immutable code revision when reporting a reproduction.

## Extend without replacing the paper evidence

Use a separate branch and output directory. State which model, protocol or evaluator changes. Keep existing tables, metadata and published release tags unchanged. Record failures and missing artifacts. Submit a focused change with tests and a provenance update; see [CONTRIBUTING.md](../CONTRIBUTING.md).
