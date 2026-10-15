"""Render the horizon and metric-tradeoff figures from released CSV artifacts.

1. fig_horizon_curves: per-forecast-step degradation of TA-ANP-noK
   (IoU/Dice, centroid error, summed-concentration error, FP/FN areas) with seed std bands.
   Source: tables/obj2_ta_anp_no_k_shape_metrics_by_timestep.csv

2. fig_paradox_scatter: global SSIM vs centroid error for the common-test-set rows,
   visualizing the last-observation paradox.
   Source: tables/obj2_acml_final_10seed_matched_baseline_metrics.csv

Run from the repository root:
    python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_paths import safe_output_directory
DEFAULT_TS_CSV = ROOT / "tables/obj2_ta_anp_no_k_shape_metrics_by_timestep.csv"
DEFAULT_MB_CSV = ROOT / "tables/obj2_acml_final_10seed_matched_baseline_metrics.csv"
DEFAULT_LAST_OBS_JSON = ROOT / "tables/last_observation_seed0_diagnostic_metrics_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon_csv", type=Path, default=DEFAULT_TS_CSV)
    parser.add_argument("--matched_baseline_csv", type=Path, default=DEFAULT_MB_CSV)
    parser.add_argument("--last_observation_json", type=Path, default=DEFAULT_LAST_OBS_JSON)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs/figures")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required released artifact not found: {path}")
    return path


def load_ts(path: Path):
    with require_file(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    t = np.array([int(r["future_timestep"]) for r in rows])
    def get(*keys: str) -> np.ndarray:
        for key in keys:
            if rows and key in rows[0] and all(row.get(key, "") != "" for row in rows):
                return np.array([float(row[key]) for row in rows])
        raise KeyError(f"None of the requested columns is complete: {keys}")

    return t, get


def fig_horizon(horizon_csv: Path, out_dir: Path) -> None:
    t, get = load_ts(horizon_csv)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.0), constrained_layout=True)

    ax = axes[0]
    for key, label, color in [("plume_iou", "IoU", "C0"), ("plume_dice", "Dice", "C1")]:
        m, s = get(f"{key}_mean"), get(f"{key}_std")
        ax.plot(t, m, "-o", ms=3, color=color, label=label)
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.2)
    ax.set_xlabel("Forecast snapshot index")
    ax.set_ylabel("Overlap")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(a) Plume overlap", fontsize=9)

    ax = axes[1]
    m, s = get("centroid_error_px_mean"), get("centroid_error_px_std")
    ax.plot(t, m, "-o", ms=3, color="C2")
    ax.fill_between(t, m - s, m + s, color="C2", alpha=0.2)
    ax.set_xlabel("Forecast snapshot index")
    ax.set_ylabel("Centroid error (px)")
    ax.set_title("(b) Displacement", fontsize=9)

    ax = axes[2]
    m = get("summed_conc_abs_error_mean", "mass_abs_error_mean")
    s = get("summed_conc_abs_error_std", "mass_abs_error_std")
    ax.plot(t, m, "-o", ms=3, color="C3")
    ax.fill_between(t, np.maximum(m - s, 1.0), m + s, color="C3", alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Forecast snapshot index")
    ax.set_ylabel("Summed-concentration absolute error (log scale)")
    ax.set_title("(c) Summed concentration", fontsize=9)

    ax = axes[3]
    fp = get("false_positive_area_px_mean")
    fn = get("false_negative_area_px_mean")
    ax.plot(t, fp, "-o", ms=3, color="C4", label="False positive")
    ax.plot(t, fn, "-s", ms=3, color="C5", label="False negative")
    ax.set_yscale("log")
    ax.set_xlabel("Forecast snapshot index")
    ax.set_ylabel("Area (px, log scale)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(d) Plume-mask error area", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.set_xticks([15, 18, 21, 24])
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_horizon_curves.{ext}"
        fig.savefig(out, dpi=300)
        print("[saved]", out)
    plt.close(fig)


def fig_paradox(matched_csv: Path, last_observation_json: Path, out_dir: Path) -> None:
    with require_file(matched_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    last_observation = json.loads(require_file(last_observation_json).read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    label_map = {
        "TA-ANP-noK (lead)": ("TA-ANP-noK (ours)", "C0", "o"),
        "TA-ANP-full": ("TA-ANP-full (ours)", "C0", "s"),
        "NeuralProcess": ("Cross-attentive NP", "C1", "o"),
        "EarthformerLite": ("EarthformerLite", "C2", "o"),
        "ACDiff": ("ACDiff", "C3", "o"),
        "DyDiff": ("DyDiff", "C3", "s"),
        "cCGAN-gan0.25": ("cCGAN $\\gamma{=}0.25$", "C4", "o"),
        "cCGAN-gan0.5": ("cCGAN $\\gamma{=}0.5$", "C4", "s"),
        "Last-observation (persistence)": ("Last-observation", "red", "X"),
    }
    # Label text anchor (data coords) per point; arrows connect text to marker so
    # no label sits on top of a marker. cCGAN 0.25/0.5 overlap, so label once.
    text_pos = {
        "TA-ANP-noK (ours)":  (0.815, 6.2),
        "TA-ANP-full (ours)": (0.815, 12.5),
        "Cross-attentive NP":     (0.792, 19.5),
        "EarthformerLite":    (0.690, 12.5),
        "ACDiff":             (0.660, 20.0),
        "DyDiff":             (0.745, 30.5),
        "cCGAN $\\gamma{=}0.25$": (0.755, 53.0),
        "Last-observation":   (0.770, 39.0),
    }
    skip_label = {"cCGAN $\\gamma{=}0.5$"}  # covered by the 0.25 combined label
    pts = {}
    for r in rows:
        name = r["model"]
        if name not in label_map:
            continue
        disp, color, marker = label_map[name]
        x = float(r["mean_ssim_log_mean"])
        y = float(r["mean_centroid_err_px_mean"])
        pts[disp] = (x, y)
        ax.scatter(x, y, c=color, marker=marker, s=55,
                   zorder=3, edgecolors="k", linewidths=0.5)
    # Deterministic last-observation values are loaded from the released
    # cache-derived diagnostic summary rather than duplicated as constants.
    lo_x = float(last_observation["global_ssim"]["mean"])
    lo_y = float(last_observation["centroid_error_px"]["mean"])
    pts["Last-observation"] = (lo_x, lo_y)
    ax.scatter(lo_x, lo_y, c="red", marker="X", s=110,
               zorder=4, edgecolors="k", linewidths=0.5)
    for disp, (x, y) in pts.items():
        if disp in skip_label:
            continue
        tx, ty = text_pos.get(disp, (x + 0.006, y + 1.2))
        lbl = "cCGAN-K $\\gamma{\\in}\\{0.25,0.5\\}$" if disp.startswith("cCGAN") else disp
        ax.annotate(lbl, xy=(x, y), xytext=(tx, ty), fontsize=9,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6,
                                    shrinkA=2, shrinkB=4))
    ax.set_xlabel("Future global SSIM (higher is better)", fontsize=10)
    ax.set_ylabel("Centroid error, px (lower is better)", fontsize=10)
    ax.set_xlim(0.645, 0.915)
    ax.set_ylim(2.0, 56.0)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_paradox_scatter.{ext}"
        fig.savefig(out, dpi=300)
        print("[saved]", out)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    args.out_dir = safe_output_directory(ROOT, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_horizon(args.horizon_csv, args.out_dir)
    fig_paradox(args.matched_baseline_csv, args.last_observation_json, args.out_dir)
