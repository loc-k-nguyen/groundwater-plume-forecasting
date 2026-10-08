# Figure sources and interpretation

The README presents a conceptual overview and two existing aggregate scientific plots. It contains no simulation field panels. Figures describe archived results; they do not introduce a new experiment. The paper PDFs and numerical evidence are unchanged.

## Research overview

`figures/research_overview.png` is the README preview; the SVG is the scalable version. `scripts/make_research_overview.py` reads `tables/primary_metric_per_seed.csv` and renders both formats. The protocol and inference schematic follow the manuscript and `src/models/ta_anp.py`.

The comparison unit is the training seed under one fixed split: ten seeds for each TA-ANP regime. Dots are individual runs; paired lines connect matching seed IDs. Diamonds and horizontal error bars show the mean and sample standard deviation, not confidence intervals. The displayed SSIM range is 0.78–0.85. The figure presents the within-family comparison, with no significance claim beyond the archived tests in `tables/obj2_acml_primary_metric_two_sided_stats.csv`.

The reader should notice that noK has higher plume SSIM in this studied design. This motivates conditioning analysis, not a conclusion that conductivity is physically unimportant. Dense observed history can retain geology information, and the conductivity-grouped split still has amplitude-rescaled training twins. Independent-geology transfer and calibrated uncertainty are untested.

## Horizon diagnostics

`figures/fig_horizon_curves.png` and its PDF are the existing aggregate plot from `tables/obj2_ta_anp_no_k_shape_metrics_by_timestep.csv`. The plotting code is `scripts/make_figures.py`. The horizontal axis is the future snapshot index, not a uniformly spaced time coordinate. Bands represent standard deviation across ten seed rows for each forecast snapshot.

The figure compares plume overlap, centroid displacement, summed-concentration absolute error and false-positive/false-negative areas. These diagnostics reveal failure modes that one similarity score can conceal. Summed concentration is a proxy; it is not a physical conservative mass-balance residual. Conclusions apply to the archived fixed split and dense-history regime.

## Similarity and displacement

`figures/fig_paradox_scatter.png` and its PDF are the existing comparison from `tables/obj2_acml_final_10seed_matched_baseline_metrics.csv`, with the last-observation diagnostic from `tables/last_observation_seed0_diagnostic_metrics_summary.json`. Its plotting code is `scripts/make_figures.py`.

The reader should compare global SSIM with centroid error rather than rank models by SSIM alone. Learned rows use the recorded seed summaries; deterministic references have no training-seed uncertainty. The last-observation diagnostic is recorded as seed 0. Different model input regimes and training-target windows prevent this chart from isolating the benefit of observation history. It is not an independent-geology benchmark.

## Rebuild without raw fields

```bash
python scripts/make_research_overview.py --out-dir outputs/reproduction/overview
python scripts/make_figures.py --out_dir outputs/reproduction/figures
python scripts/build_manifest.py --check
```

Use the CPU dependencies in `requirements-cpu-lock.txt` and CPU PyTorch as documented in the README. The final manifest records the displayed images and script hashes. Generated outputs stay under ignored `outputs/`; archived numerical files are never rewritten.
