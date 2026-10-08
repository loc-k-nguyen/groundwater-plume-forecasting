# Headline Significance Test: TA-ANP-noK vs Neural Process

Metric: future plume-region SSIM (mean over 170 test files x 10 future steps), per seed.
Test: seed-aligned across 10 seeds (0..9).

Sources (per-seed, traceable):
- noK: `obj2_ta_anp_10seed_physical_metrics.csv`, row `ta_anp_no_k / future_plume_ssim`, seed_values column.
- Neural Process: per-seed future metric reports, key `mean_plume_ssim_log`.

| seed | noK | NP | diff |
|---|---|---|---|
| 0 | 0.8305 | 0.7504 | +0.0802 |
| 1 | 0.8315 | 0.7599 | +0.0716 |
| 2 | 0.8399 | 0.7741 | +0.0657 |
| 3 | 0.8300 | 0.7526 | +0.0773 |
| 4 | 0.8372 | 0.7526 | +0.0845 |
| 5 | 0.8235 | 0.7801 | +0.0434 |
| 6 | 0.8298 | 0.7231 | +0.1067 |
| 7 | 0.8308 | 0.7642 | +0.0666 |
| 8 | 0.8330 | 0.7392 | +0.0938 |
| 9 | 0.8403 | 0.7825 | +0.0578 |

- noK = 0.8326 +/- 0.0051 ; NP = 0.7579 +/- 0.0185 ; mean seed-aligned diff = +0.0748
- All 10 seeds favor noK.
- Wilcoxon signed-rank, one-sided (noK > NP): W = 55.0, p = 0.00098 (minimum attainable at n=10).
- Paired t-test (two-sided): t = 13.07, p = 3.7e-07.
- Effect size: Cohen's d (paired) = 4.13.

Used in manuscript Results 7.1 (matched baseline comparison): "consistent across all ten seeds
and statistically significant (seed-aligned Wilcoxon one-sided p=0.001; seed-aligned t-test p<1e-6,
Cohen's d=4.1)."
