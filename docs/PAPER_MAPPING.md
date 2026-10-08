# Manuscript and code mapping

The last submission identified in local records is manuscript v14b. Manuscript v15 is the October 3 correction draft; v16 is the October 8 conditional camera-ready preparation. Neither local draft is evidence of a portal upload. Software snapshot identifiers are separate from these manuscript versions, and a final software tag is intentionally absent.

## Evidence mapping

| Manuscript content | Released evidence or code |
|---|---|
| Main model and noK/full/FiLM regimes | `src/models/ta_anp.py`, `src/models/neural_process.py`, `configs/ta_anp_*.json`, `scripts/train_ta_anp.py` |
| Temporal supervision and inference differences | `metadata/training_protocols.csv`, `metadata/param_split_fixed.json`, `src/data/dataset_interp.py` |
| Main common-test-set comparison | `tables/obj2_acml_final_10seed_matched_baseline_metrics.csv` |
| Component ablations | `tables/obj2_ta_anp_10seed_ablation_summary.csv`, `configs/ta_anp_no_*.json` |
| Primary two-sided tests | `tables/primary_metric_per_seed.csv`, `tables/obj2_acml_primary_metric_two_sided_stats.csv`, `scripts/verify_revision_evidence.py` |
| Plume and full-field diagnostics | `tables/obj2_acml_ta_anp_full_vs_nok_shape_metrics.csv`, `scripts/compute_plume_metrics.py` |
| Pairing disclosure, Supplementary S6 | `metadata/pairing_and_screening_audit.json` |
| Screening, Supplementary S7 | `tables/obj2_acml_screened_all_metrics.csv`, `tables/obj2_acml_screened_all_metrics_tests.csv`, threshold summaries |
| Prior predictive sampling, Supplementary S8 | `tables/uq_ta_anp_no_k.csv`, `tables/uq_ta_anp_no_k.json`, `tables/uq_decompose_no_k.json` |
| Grouped comparison and twins, Supplementary S10 | `metadata/param_split_grouped_k_v5.json`, `metadata/conductivity_twin_audit.json`, `tables/obj2_acml_grouped_split.json` |

Main-split evaluation contains 170 test realizations; the grouped split contains 180. The grouped split is byte-disjoint, not statistically independent. Nominal identifiers remain intact and do not certify the generating parameters of the exceptional rerun. Later independent-geology campaigns and research extensions are excluded.

The fixed-split definition, model code and source metrics are preserved. New preparation code checks and aggregates those artifacts without altering validated numerical sources. `docs/source_snapshot.json` records source and copied SHA-256 values, with restricted-field transformations explicitly listed. `MANIFEST.csv` covers the final included file set. No raw field, simulator setup or checkpoint is licensed or redistributed.

The included configuration files are representative archived settings, not a claim to include every seed's full executable launch environment. Summary checks do not replace cache-level or retraining reproduction. Formal acceptance, final reference metadata and a public code URL must be reconciled with the authors' final portal version before tagging.
