"""Compute plume shape and physical metrics from validated prediction caches.

The script reads existing prediction caches only. It does not run model
inference or require GPU. Metrics are computed in physical concentration space
after inverting log10(C + eps) as C = max(10**log_c - eps, 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.centroid_error import compute_plume_centroid_error
from src.eval.plume_ssim import compute_global_ssim, compute_plume_ssim


PLUME_THRESH = 1e-8
EPS_C = 1e-12
PLUME_MIN_PIXELS = 64
FUTURE_TIMESTEPS = list(range(15, 25))


@dataclass
class MetricSpec:
    name: str
    direction: str
    unit: str
    space: str
    status: str = "computed"
    reason: str = ""


METRIC_SPECS = {
    "global_ssim": MetricSpec("global_ssim", "higher_is_better", "ssim", "log10_C"),
    "gt_roi_plume_ssim": MetricSpec("gt_roi_plume_ssim", "higher_is_better", "ssim", "log10_C"),
    "union_roi_plume_ssim": MetricSpec(
        "union_roi_plume_ssim",
        "higher_is_better",
        "ssim",
        "log10_C",
        "blocked",
        "Not recomputed in this pass; requires a dedicated SSIM/ROI pass over large arrays.",
    ),
    "plume_iou": MetricSpec("plume_iou", "higher_is_better", "ratio", "physical"),
    "plume_dice": MetricSpec("plume_dice", "higher_is_better", "ratio", "physical"),
    "plume_area_rel_error": MetricSpec("plume_area_rel_error", "lower_is_better", "relative_error", "physical"),
    "false_positive_area_px": MetricSpec("false_positive_area_px", "lower_is_better", "pixels", "physical"),
    "false_negative_area_px": MetricSpec("false_negative_area_px", "lower_is_better", "pixels", "physical"),
    "false_positive_area_rel": MetricSpec("false_positive_area_rel", "lower_is_better", "relative_to_gt_area", "physical"),
    "false_negative_area_rel": MetricSpec("false_negative_area_rel", "lower_is_better", "relative_to_gt_area", "physical"),
    "plume_miss_rate": MetricSpec("plume_miss_rate", "lower_is_better", "ratio", "physical"),
    "centroid_error_px": MetricSpec("centroid_error_px", "lower_is_better", "pixels", "physical_weighted"),
    "peak_concentration_abs_error": MetricSpec("peak_concentration_abs_error", "lower_is_better", "concentration", "physical"),
    "extent_front_error_px": MetricSpec("extent_front_error_px", "lower_is_better", "pixels", "physical_mask_bbox"),
    "summed_conc_abs_error": MetricSpec("summed_conc_abs_error", "lower_is_better", "summed_concentration", "physical"),
    "summed_conc_rel_error": MetricSpec("summed_conc_rel_error", "lower_is_better", "relative_error", "physical"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute plume shape and physical metrics from prediction caches.")
    parser.add_argument("--variant", default="ta_anp_no_k",
                        help="variant name stored in the cache metadata; used to validate caches.")
    parser.add_argument("--cache_root", default=None,
                        help="directory holding seed*/pred_future.npz; defaults to predictions/<variant>.")
    parser.add_argument("--out_root", default="reports")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--plume_thresh", type=float, default=PLUME_THRESH)
    parser.add_argument("--eps_c", type=float, default=EPS_C)
    parser.add_argument("--min_pixels", type=int, default=PLUME_MIN_PIXELS)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="accept smaller reviewer/smoke caches while retaining schema and timestep checks",
    )
    args = parser.parse_args()
    if args.cache_root is None:
        args.cache_root = f"predictions/{args.variant}"
    return args


def log_to_phys(log_c: np.ndarray, eps_c: float) -> np.ndarray:
    return np.clip(np.power(10.0, log_c.astype(np.float64)) - float(eps_c), 0.0, None)


def extent_error(gt_mask: np.ndarray, pred_mask: np.ndarray, min_pixels: int) -> float:
    if int(gt_mask.sum()) < int(min_pixels) or int(pred_mask.sum()) < int(min_pixels):
        return float("nan")
    gt_rows, gt_cols = np.where(gt_mask)
    pr_rows, pr_cols = np.where(pred_mask)
    diffs = [
        abs(float(gt_rows.min()) - float(pr_rows.min())),
        abs(float(gt_rows.max()) - float(pr_rows.max())),
        abs(float(gt_cols.min()) - float(pr_cols.min())),
        abs(float(gt_cols.max()) - float(pr_cols.max())),
    ]
    return float(np.mean(diffs))


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.std(ddof=1)) if arr.size > 1 else float("nan")


def row_metric_values(sample_metrics: dict[str, float]) -> dict[str, float]:
    return {name: sample_metrics.get(name, float("nan")) for name in METRIC_SPECS}


def validate_cache(path: Path, seed: int, variant: str, allow_partial: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"seed": seed, "path": str(path), "exists": path.exists()}
    if not path.exists():
        row["valid"] = False
        row["reason"] = "missing cache file"
        return row
    with np.load(path) as data:
        required = ["y_pred", "y_true", "timestep_indices", "model_name", "variant", "seed"]
        missing = [key for key in required if key not in data.files]
        if missing:
            row.update({"valid": False, "reason": f"missing keys: {missing}"})
            return row
        y_pred = data["y_pred"]
        y_true = data["y_true"]
        timesteps = data["timestep_indices"].astype(int).tolist()
        row.update(
            {
                "keys": sorted(data.files),
                "y_pred_shape": list(y_pred.shape),
                "y_true_shape": list(y_true.shape),
                "sample_count": int(y_pred.shape[0]) if y_pred.ndim >= 1 else None,
                "timestep_count": int(y_pred.shape[1]) if y_pred.ndim >= 2 else None,
                "timesteps": timesteps,
                "y_pred_has_nan": bool(np.isnan(y_pred).any()),
                "y_true_has_nan": bool(np.isnan(y_true).any()),
                "y_pred_all_finite": bool(np.isfinite(y_pred).all()),
                "y_true_all_finite": bool(np.isfinite(y_true).all()),
                "y_pred_min": float(np.min(y_pred)),
                "y_pred_max": float(np.max(y_pred)),
                "y_true_min": float(np.min(y_true)),
                "y_true_max": float(np.max(y_true)),
                "model_name": str(data["model_name"]),
                "variant": str(data["variant"]),
                "seed_meta": int(data["seed"]),
                "protocol": str(data["protocol"]) if "protocol" in data.files else "",
            }
        )
        shape_valid = bool(
            y_pred.ndim == 4
            and y_true.shape == y_pred.shape
            and y_pred.shape[0] > 0
            and y_pred.shape[1:] == (10, 600, 400)
        )
        if not allow_partial:
            shape_valid = shape_valid and y_pred.shape[0] == 170
        row["valid"] = bool(
            shape_valid
            and timesteps == FUTURE_TIMESTEPS
            and not row["y_pred_has_nan"]
            and not row["y_true_has_nan"]
            and row["y_pred_all_finite"]
            and row["y_true_all_finite"]
            and row["variant"] == variant
            and row["seed_meta"] == seed
        )
        return row


def compute_seed_metrics(path: Path, seed: int, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    with np.load(path) as data:
        y_pred = data["y_pred"]
        y_true = data["y_true"]
        sample_ids = data["sample_id"].astype(str).tolist() if "sample_id" in data.files else [str(i) for i in range(y_pred.shape[0])]
        param_ids = data["param_id"].astype(str).tolist() if "param_id" in data.files else [""] * y_pred.shape[0]
        real_ids = data["real_id"].astype(str).tolist() if "real_id" in data.files else [""] * y_pred.shape[0]
        n_samples, n_steps = int(y_pred.shape[0]), int(y_pred.shape[1])

        for start in range(0, n_samples, int(args.batch_size)):
            end = min(n_samples, start + int(args.batch_size))
            pred_phys = log_to_phys(y_pred[start:end], args.eps_c)
            true_phys = log_to_phys(y_true[start:end], args.eps_c)
            pred_mask = pred_phys > float(args.plume_thresh)
            true_mask = true_phys > float(args.plume_thresh)

            for local_i, sample_idx in enumerate(range(start, end)):
                for t_idx in range(n_steps):
                    gt_mask = true_mask[local_i, t_idx]
                    pr_mask = pred_mask[local_i, t_idx]
                    gt = true_phys[local_i, t_idx]
                    pr = pred_phys[local_i, t_idx]
                    gt_log = y_true[sample_idx, t_idx]
                    pr_log = y_pred[sample_idx, t_idx]

                    gt_area = int(gt_mask.sum())
                    pr_area = int(pr_mask.sum())
                    inter = int(np.logical_and(gt_mask, pr_mask).sum())
                    union = int(np.logical_or(gt_mask, pr_mask).sum())
                    fp = int(np.logical_and(pr_mask, ~gt_mask).sum())
                    fn = int(np.logical_and(gt_mask, ~pr_mask).sum())
                    valid_gt = gt_area >= int(args.min_pixels)

                    # Use the shared centroid implementation so every script applies
                    # the same plume-centroid convention and empty-prediction handling.
                    centroid_err, _ = compute_plume_centroid_error(
                        gt_phys=gt,
                        pred_phys=pr,
                        thresh=float(args.plume_thresh),
                        min_pixels=int(args.min_pixels),
                    )

                    gt_mass = float(gt.sum())
                    pr_mass = float(pr.sum())
                    summed_conc_abs = abs(pr_mass - gt_mass)
                    plume_ssim, _ = compute_plume_ssim(
                        gt_phys=gt,
                        pred_log=pr_log,
                        eps_c=float(args.eps_c),
                        thresh=float(args.plume_thresh),
                        min_pixels=int(args.min_pixels),
                    )
                    sample_rows.append(
                        {
                            "seed": seed,
                            "sample_index": sample_idx,
                            "sample_id": sample_ids[sample_idx],
                            "param_id": param_ids[sample_idx],
                            "real_id": real_ids[sample_idx],
                            "future_timestep": FUTURE_TIMESTEPS[t_idx],
                            "global_ssim": compute_global_ssim(gt_log, pr_log),
                            "gt_roi_plume_ssim": plume_ssim,
                            "gt_area_px": gt_area,
                            "pred_area_px": pr_area,
                            "plume_iou": float(inter / union) if union > 0 else float("nan"),
                            "plume_dice": float((2 * inter) / (gt_area + pr_area)) if (gt_area + pr_area) > 0 else float("nan"),
                            "plume_area_rel_error": float(abs(pr_area - gt_area) / gt_area) if valid_gt else float("nan"),
                            "false_positive_area_px": fp,
                            "false_negative_area_px": fn,
                            "false_positive_area_rel": float(fp / gt_area) if valid_gt else float("nan"),
                            "false_negative_area_rel": float(fn / gt_area) if valid_gt else float("nan"),
                            "plume_miss_rate": float(fn / gt_area) if valid_gt else float("nan"),
                            "centroid_error_px": float(centroid_err),
                            "peak_concentration_abs_error": abs(float(pr.max()) - float(gt.max())),
                            "extent_front_error_px": extent_error(gt_mask, pr_mask, args.min_pixels),
                            "summed_conc_abs_error": summed_conc_abs,
                            "summed_conc_rel_error": float(summed_conc_abs / gt_mass) if gt_mass > 0.0 else float("nan"),
                        }
                    )

    per_seed: dict[str, Any] = {"seed": seed, "n_samples": n_samples, "n_timestep_pairs": len(sample_rows)}
    for metric in METRIC_SPECS:
        per_seed[metric] = safe_mean(row.get(metric, float("nan")) for row in sample_rows)

    by_timestep: list[dict[str, Any]] = []
    for timestep in FUTURE_TIMESTEPS:
        rows_t = [row for row in sample_rows if row["future_timestep"] == timestep]
        out = {"seed": seed, "future_timestep": timestep, "n_pairs": len(rows_t)}
        for metric in METRIC_SPECS:
            out[metric] = safe_mean(row.get(metric, float("nan")) for row in rows_t)
        by_timestep.append(out)

    return per_seed, by_timestep, sample_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in METRIC_SPECS.items():
        values = [float(row[name]) for row in per_seed_rows if name in row and np.isfinite(float(row[name]))]
        rows.append(
            {
                "metric": name,
                "status": spec.status,
                "reason": spec.reason,
                "direction": spec.direction,
                "unit": spec.unit,
                "space": spec.space,
                "n_valid_seeds": len(values),
                "mean": safe_mean(values),
                "std": safe_std(values),
                "standard_error": safe_std(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan"),
            }
        )
    return rows


def build_by_timestep_summary(by_seed_timestep: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestep in FUTURE_TIMESTEPS:
        seed_rows = [row for row in by_seed_timestep if row["future_timestep"] == timestep]
        out: dict[str, Any] = {"future_timestep": timestep, "n_seed_rows": len(seed_rows)}
        for name, spec in METRIC_SPECS.items():
            values = [float(row[name]) for row in seed_rows if name in row and np.isfinite(float(row[name]))]
            out[f"{name}_mean"] = safe_mean(values)
            out[f"{name}_std"] = safe_std(values)
            out[f"{name}_status"] = spec.status
        rows.append(out)
    return rows


def build_selection_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in sample_rows:
        by_sample.setdefault((int(row["seed"]), int(row["sample_index"])), []).append(row)

    agg: list[dict[str, Any]] = []
    for (seed, sample_idx), rows in by_sample.items():
        out = {
            "seed": seed,
            "sample_index": sample_idx,
            "sample_id": rows[0]["sample_id"],
            "param_id": rows[0]["param_id"],
            "real_id": rows[0]["real_id"],
            "plume_iou": safe_mean(r["plume_iou"] for r in rows),
            "centroid_error_px": safe_mean(r["centroid_error_px"] for r in rows),
            "summed_conc_abs_error": safe_mean(r["summed_conc_abs_error"] for r in rows),
            "false_positive_area_px": safe_mean(r["false_positive_area_px"] for r in rows),
            "false_negative_area_px": safe_mean(r["false_negative_area_px"] for r in rows),
        }
        agg.append(out)

    selections: list[dict[str, Any]] = []

    def add(label: str, row: dict[str, Any]) -> None:
        selected = dict(row)
        selected["selection_label"] = label
        selections.append(selected)

    finite_iou = [r for r in agg if np.isfinite(r["plume_iou"])]
    if finite_iou:
        sorted_iou = sorted(finite_iou, key=lambda r: r["plume_iou"])
        add("worst_iou", sorted_iou[0])
        add("median_iou", sorted_iou[len(sorted_iou) // 2])
        add("best_iou", sorted_iou[-1])
    for label, key in [
        ("high_centroid_error", "centroid_error_px"),
        ("high_false_positive_area", "false_positive_area_px"),
        ("high_false_negative_area", "false_negative_area_px"),
        ("high_summed_conc_error", "summed_conc_abs_error"),
    ]:
        finite = [r for r in agg if np.isfinite(r[key])]
        if finite:
            add(label, max(finite, key=lambda r: r[key]))
    return selections



def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    out_root = Path(args.out_root)
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    validation_rows, per_seed_rows, by_seed_timestep_rows, all_sample_rows = [], [], [], []
    for seed in args.seeds:
        cache_path = cache_root / f"seed{seed}" / "pred_future.npz"
        validation = validate_cache(cache_path, seed, args.variant, args.allow_partial)
        validation_rows.append(validation)
        if not validation.get("valid", False):
            continue
        per_seed, by_timestep, sample_rows = compute_seed_metrics(cache_path, seed, args)
        per_seed_rows.append(per_seed)
        by_seed_timestep_rows.extend(by_timestep)
        all_sample_rows.extend(sample_rows)

    validation_payload = {
        "all_valid": all(bool(r.get("valid", False)) for r in validation_rows),
        "metric_threshold": args.plume_thresh,
        "eps_c": args.eps_c,
        "expected_shape": [170, 10, 600, 400],
        "expected_timesteps": FUTURE_TIMESTEPS,
        "rows": validation_rows,
    }
    (tables_dir / "plume_shape_cache_validation.json").write_text(
        json.dumps(validation_payload, indent=2), encoding="utf-8")

    summary_rows = build_summary(per_seed_rows)
    by_timestep_rows = build_by_timestep_summary(by_seed_timestep_rows)
    selection_rows = build_selection_rows(all_sample_rows)

    write_csv(tables_dir / "plume_shape_metrics_per_seed.csv", per_seed_rows)
    write_csv(tables_dir / "plume_shape_metrics_summary.csv", summary_rows)
    write_csv(tables_dir / "plume_shape_metrics_by_timestep.csv", by_timestep_rows)
    write_csv(tables_dir / "plume_qualitative_selection.csv", selection_rows)
    print("Wrote plume shape metrics to", tables_dir)


if __name__ == "__main__":
    main()
