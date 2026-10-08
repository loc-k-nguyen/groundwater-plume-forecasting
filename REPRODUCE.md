# Reproduction

Run all commands from the repository root. The CPU quick start in README uses Python 3.9, PyTorch 2.1.0 CPU and the pinned analysis requirements. It is an inspection environment, distinct from the recorded GPU training environment; it does not certify numerical equivalence of retraining on another stack.

## Data-free verification

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
python scripts/validate_release_artifacts.py
python scripts/verify_revision_evidence.py
python scripts/reproduce_summaries.py --out-dir outputs/reproduction
python scripts/make_figures.py --out_dir outputs/reproduction/figures
python scripts/build_manifest.py --check
```

The summary script recalculates means and sample standard deviations from released seed records and exact two-sided Wilcoxon/paired t tests for the grouped split. It verifies the stored grouped statistics within explicitly stated rounding tolerances and retains the uncalibrated uncertainty and conductivity-twin findings. Screening outputs are read from archived results, not recomputed from absent prediction caches. The revision-evidence script recomputes all six primary statistical comparisons from the released per-seed values and checks the artifact-map hashes.

Generated output paths must be outside `tables/` and `metadata/`. No result source should be overwritten. Manifest generation excludes generated output, virtual environments, cache directories and Git metadata.

## Entry-point checks

```bash
python scripts/train_ta_anp.py --help
python scripts/export_prediction_cache.py --help
python scripts/compute_plume_metrics.py --help
python scripts/run_naive_baselines.py --help
python scripts/paired_significance_tests.py --help
python scripts/make_figures.py --help
python scripts/build_manifest.py --help
python scripts/validate_release_artifacts.py --help
```

These are CLI checks, not training or full-field evaluation. No cluster credentials are required.

## Restricted-data stages

Substantive training needs the frozen simulation fields, fixed split and recorded normalization. The noK configuration uses `use_k=false`, `use_flow=true`, 150 epochs, patch size 320, batch size 1 with gradient accumulation 16, AdamW `lr=1e-4`, zero weight decay and the archived loss coefficients. Do not reinterpret indices as uniformly spaced days. `C[0:14]` in the protocol record denotes inclusive indices 0–14, not Python slice syntax.

Model selection and evaluation must follow the paper's target windows. The earlier-target references are not controlled measurements of the benefit of history. Checkpoints, complete reference-model trainers, prediction caches and raw fields are absent. Their regeneration and full numerical reproduction have not been tested in this private preparation. No new training, fresh-seed simulation or journal-extension result is included.

## Fresh clone

Clone the private URL using your own authenticated Git access, install the CPU requirements, and run the data-free commands above. A clean checkout should remain clean after verification because generated outputs are ignored. Public access and a final release tag are separate gates.
