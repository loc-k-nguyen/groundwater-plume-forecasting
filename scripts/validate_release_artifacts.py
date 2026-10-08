"""Validate manuscript-facing CSV schemas without requiring model data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required release artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_matched_table(path: Path) -> None:
    rows = read_csv(path)
    expected_seeds = set(range(10))
    for row in rows:
        if not row.get("n_seeds"):
            continue
        seeds = {int(value) for value in row["seeds"].split(";") if value != ""}
        if int(row["n_seeds"]) != 10 or seeds != expected_seeds:
            raise ValueError(f"Incomplete learned row in {path.name}: {row.get('model')}")
        if row.get("model") == "TA-ANP-no-transport-loss":
            raise ValueError("Use the evidence-aligned ablation label TA-ANP-no-auxiliary-loss.")


def validate_stats(path: Path) -> None:
    rows = read_csv(path)
    required = {
        "wilcoxon_one_sided_p",
        "wilcoxon_two_sided_p",
        "seed_aligned_t_one_sided_p",
        "seed_aligned_t_two_sided_p",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Statistical-test directions are not explicit in {path.name}.")


def validate_timestep_metrics(path: Path) -> None:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No rows found in {path.name}.")
    columns = set(rows[0])
    current = {"summed_conc_abs_error_mean", "summed_conc_abs_error_std"}
    archived = {"mass_abs_error_mean", "mass_abs_error_std"}
    if not (current.issubset(columns) or archived.issubset(columns)):
        raise ValueError(f"No supported summed-concentration columns found in {path.name}.")


def validate_ablation_config(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required release configuration is missing: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("lambda_plume", "lambda_centroid", "lambda_mass"):
        if config.get(key) != 0.0:
            raise ValueError(f"{path.name} must set {key}=0.0 to match the archived ablation.")


def main() -> None:
    root = parse_args().root.resolve()
    tables = root / "tables"
    validate_matched_table(tables / "obj2_acml_final_10seed_matched_baseline_metrics.csv")
    validate_stats(tables / "obj2_acml_primary_metric_stats.csv")
    validate_timestep_metrics(tables / "obj2_ta_anp_no_k_shape_metrics_by_timestep.csv")
    validate_ablation_config(root / "configs" / "ta_anp_no_auxiliary_loss.json")
    print("PASS: released manuscript-facing artifact schemas are consistent.")


if __name__ == "__main__":
    main()
