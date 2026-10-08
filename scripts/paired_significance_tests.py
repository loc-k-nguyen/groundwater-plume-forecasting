"""Run seed-aligned tests between two per-seed metric tables.

The two CSV files must contain one value per seed for the selected columns.
The script reports Wilcoxon and paired t tests plus the standardized mean
within-seed difference. It does not perform model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left_csv", required=True)
    parser.add_argument("--right_csv", required=True)
    parser.add_argument("--seed_col", default="seed")
    parser.add_argument("--left_value_col", required=True)
    parser.add_argument("--right_value_col", required=True)
    parser.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided")
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()


def load_values(path: str, seed_col: str, value_col: str) -> dict[int, float]:
    values: dict[int, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get(seed_col, "") == "" or row.get(value_col, "") == "":
                continue
            values[int(row[seed_col])] = float(row[value_col])
    return values


def compute_statistics(
    left: dict[int, float],
    right: dict[int, float],
    alternative: str,
) -> dict[str, object]:
    seeds = sorted(set(left) & set(right))
    if len(seeds) < 2:
        raise ValueError("At least two seed-aligned observations are required.")

    x = np.asarray([left[s] for s in seeds], dtype=np.float64)
    y = np.asarray([right[s] for s in seeds], dtype=np.float64)
    diff = x - y
    wilcoxon_requested = wilcoxon(diff, alternative=alternative)
    wilcoxon_two_sided = wilcoxon(diff, alternative="two-sided")
    t_requested = ttest_rel(x, y, alternative=alternative)
    t_two_sided = ttest_rel(x, y, alternative="two-sided")
    diff_std = float(diff.std(ddof=1))
    return {
        "seeds": seeds,
        "n_seed_aligned": len(seeds),
        "left_mean": float(x.mean()),
        "left_std": float(x.std(ddof=1)),
        "right_mean": float(y.mean()),
        "right_std": float(y.std(ddof=1)),
        "mean_difference_left_minus_right": float(diff.mean()),
        "left_better_seed_count": int((diff > 0).sum()),
        "wilcoxon_statistic_requested": float(wilcoxon_requested.statistic),
        "wilcoxon_p_requested": float(wilcoxon_requested.pvalue),
        "wilcoxon_statistic_two_sided": float(wilcoxon_two_sided.statistic),
        "wilcoxon_p_two_sided": float(wilcoxon_two_sided.pvalue),
        "paired_t_statistic": float(t_requested.statistic),
        "paired_t_p_requested": float(t_requested.pvalue),
        "paired_t_p_two_sided": float(t_two_sided.pvalue),
        "within_seed_cohens_d": float(diff.mean() / diff_std) if diff_std > 0 else float("inf"),
        "alternative": alternative,
    }


def main() -> None:
    args = parse_args()
    left = load_values(args.left_csv, args.seed_col, args.left_value_col)
    right = load_values(args.right_csv, args.seed_col, args.right_value_col)
    payload = compute_statistics(left, right, args.alternative)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
