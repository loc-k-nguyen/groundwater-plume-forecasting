"""Rebuild paper summary checks from included seed values, without training or raw fields."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_paths import safe_output_directory

def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def build() -> dict:
    primary = csv_rows(ROOT/'tables/primary_metric_per_seed.csv')
    headline = {}
    for family in sorted({r['family'] for r in primary}):
        subset = [r for r in primary if r['family'] == family]
        assert {int(r['seed']) for r in subset} == set(range(10))
        headline[family] = {key: {'mean': float(np.mean([float(r[key]) for r in subset])),
                                  'sd': float(np.std([float(r[key]) for r in subset], ddof=1))}
                            for key in ['future_plume_ssim', 'global_ssim']}
    source = json.loads((ROOT/'tables/obj2_acml_grouped_split.json').read_text())
    grouped = {}
    for key, row in source['metrics'].items():
        x, y = np.array(row['per_seed_noK']), np.array(row['per_seed_full'])
        assert len(x) == len(y) == 10
        p = float(wilcoxon(x, y, alternative='two-sided', method='exact').pvalue)
        t_p = float(ttest_rel(x, y).pvalue)
        assert np.isclose(p, row['wilcoxon_p'], atol=1e-12, rtol=0)
        assert np.isclose(t_p, row['paired_t_p'], atol=1e-12, rtol=0)
        assert np.isclose(x.mean(), row['noK_mean'], atol=1e-12, rtol=0)
        assert np.isclose(x.std(ddof=1), row['noK_sd'], atol=1e-12, rtol=0)
        assert np.isclose(y.mean(), row['full_mean'], atol=1e-12, rtol=0)
        assert np.isclose(y.std(ddof=1), row['full_sd'], atol=1e-12, rtol=0)
        grouped[key] = {'noK_mean': float(x.mean()), 'noK_sd': float(x.std(ddof=1)),
                        'full_mean': float(y.mean()), 'full_sd': float(y.std(ddof=1)),
                        'wilcoxon_two_sided_p': p, 'paired_t_two_sided_p': t_p}
    uq = [row for row in csv_rows(ROOT/'tables/uq_ta_anp_no_k.csv') if row['seed'].isdigit()]
    assert {int(r['seed']) for r in uq} == set(range(10))
    uq_summary = {key: {'mean': float(np.mean([float(r[key]) for r in uq])),
                        'sd': float(np.std([float(r[key]) for r in uq], ddof=1))}
                  for key in ['coverage', 'sharpness', 'sample_std_mean', 'risk_brier']}
    assert round(uq_summary['coverage']['mean'], 4) == 0.3881
    audit = json.loads((ROOT/'metadata/conductivity_twin_audit.json').read_text())
    assert audit['test']['distinct_fields'] == 31
    assert audit['test']['with_r_ge_0999_in_train'] == 30
    assert audit['val']['distinct_fields'] == audit['val']['with_r_ge_0999_in_train'] == 15
    return {'primary_seed_summaries': headline, 'grouped_split': grouped,
            'prior_predictive': uq_summary,
            'conductivity_twins': {'test_distinct': 31, 'test_twinned': 30, 'val_distinct': 15, 'val_twinned': 15},
            'screening_archived_summaries': csv_rows(ROOT/'tables/obj2_acml_screened_all_metrics.csv'),
            'scope': 'Summary reproduction only. Raw-field/cache screening and full retraining are not performed.'}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    out = safe_output_directory(ROOT, args.out_dir)
    result = build()
    out.mkdir(parents=True, exist_ok=True)
    (out/'paper_summary_checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('PASS: primary summaries, grouped seed statistics, uncertainty and twin evidence')

if __name__ == '__main__':
    main()
