"""Verify the revised primary statistics from released per-seed evidence."""
from pathlib import Path
import csv
import hashlib
import math
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    records = read_csv(ROOT / "tables/primary_metric_per_seed.csv")
    values = {(row["family"], int(row["seed"])): row for row in records}
    assert len(values) == len(records) == 30, "Duplicate or missing records"
    names = ["TA-ANP-noK", "TA-ANP-full", "cross-attentive NP"]
    assert set(values) == {(name, seed) for name in names for seed in range(10)}
    stats = read_csv(ROOT / "tables/obj2_acml_primary_metric_two_sided_stats.csv")
    assert len(stats) == 6
    for row in stats:
        left, right = row["comparison"].split(" vs ")
        metric = row["metric"]
        x = np.array([float(values[left, seed][metric]) for seed in range(10)])
        y = np.array([float(values[right, seed][metric]) for seed in range(10)])
        delta = x - y
        checks = {
            "left_mean": (x.mean(), 1e-9),
            "right_mean": (y.mean(), 1e-9),
            "mean_difference": (delta.mean(), 1e-9),
            "left_better_seed_count": ((delta > 0).sum(), 0),
            "wilcoxon_two_sided_p": (wilcoxon(x, y, alternative="two-sided", method="exact").pvalue, 1e-12),
            "t_two_sided_p": (ttest_rel(x, y, alternative="two-sided").pvalue, 1e-10),
            "within_seed_cohens_d": (delta.mean() / delta.std(ddof=1), 5.1e-5),
        }
        assert row["seeds"] == ";".join(map(str, range(10)))
        assert int(row["n_seed_aligned"]) == 10
        for key, (actual, tolerance) in checks.items():
            assert math.isclose(float(actual), float(row[key]), rel_tol=0, abs_tol=tolerance), (row["comparison"], metric, key, actual, row[key])
        print("PASS:", row["comparison"], metric)
    for row in read_csv(ROOT / "ARTIFACT_MAP.csv"):
        path = ROOT / row["path"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    print("PASS: all six primary comparisons and all artifact-map hashes")


if __name__ == "__main__":
    main()

