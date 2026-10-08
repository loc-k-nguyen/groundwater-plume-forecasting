"""Generate deterministic forecasting baselines in the prediction-cache schema."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as image_shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_timecond_patch import log10_C
from src.data.io import flatten_param_folders, load_param_split


OBSERVED = list(range(15))
FUTURE = list(range(15, 25))
Y_MIN, Y_MAX, EPS_C = -12.0, 2.0, 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--baseline", choices=["last_observation", "linear", "centroid_shift"], required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def sample_ids(path: str) -> tuple[str, str, str]:
    normalized = Path(path).as_posix()
    param = re.search(r"(param_[^/]+)", normalized)
    real = re.search(r"(real_[^/.]+)", normalized)
    param_id = param.group(1) if param else ""
    real_id = real.group(1) if real else Path(path).stem
    return f"{param_id}/{real_id}", param_id, real_id


def centroid(field: np.ndarray) -> np.ndarray:
    total = float(field.sum())
    if total <= 1e-8:
        return np.asarray([field.shape[0] / 2.0, field.shape[1] / 2.0])
    yy, xx = np.indices(field.shape)
    return np.asarray([(yy * field).sum() / total, (xx * field).sum() / total])


def predict(obs: np.ndarray, observed_days: np.ndarray, future_days: np.ndarray, baseline: str) -> np.ndarray:
    if baseline == "last_observation":
        return np.repeat(obs[-1][None], len(future_days), axis=0)
    if baseline == "linear":
        x = observed_days.astype(np.float64)
        centered = x - x.mean()
        values = obs.reshape(len(obs), -1).astype(np.float64)
        slope = (centered[:, None] * (values - values.mean(0))).sum(0) / ((centered**2).sum() + 1e-12)
        intercept = values.mean(0) - slope * x.mean()
        result = np.stack([np.clip(slope * day + intercept, Y_MIN, Y_MAX) for day in future_days])
        return result.reshape(len(future_days), *obs.shape[1:]).astype(np.float32)

    last_phys = np.maximum(10.0 ** obs[-1].astype(np.float64) - EPS_C, 0.0)
    recent = np.stack([centroid(np.maximum(10.0 ** frame.astype(np.float64) - EPS_C, 0.0)) for frame in obs[-3:]])
    velocity = np.diff(recent, axis=0).mean(0)
    day_scale = max(float(observed_days[-1] - observed_days[-2]), 1e-6)
    forecasts = []
    for day in future_days:
        offset = velocity * ((float(day) - float(observed_days[-1])) / day_scale)
        shifted = image_shift(last_phys, shift=tuple(offset), order=1, mode="constant", cval=0.0)
        forecasts.append(np.clip(np.log10(shifted + EPS_C), Y_MIN, Y_MAX))
    return np.asarray(forecasts, dtype=np.float32)


def main() -> None:
    args = parse_args()
    _, _, test_params = load_param_split(args.split_json)
    files = flatten_param_folders(args.data_root, test_params)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(
            "No test .npz files were found. Check --data_root and --split_json."
        )

    predictions, targets, ids, params, reals = [], [], [], [], []
    for path in files:
        with np.load(path) as data:
            concentration = data["C"].astype(np.float32)
            days = data["times"].astype(np.float32)
        if concentration.shape[0] < 25:
            continue
        log_concentration = log10_C(concentration, EPS_C)
        predictions.append(predict(log_concentration[OBSERVED], days[OBSERVED], days[FUTURE], args.baseline))
        targets.append(log_concentration[FUTURE])
        sample, param, real = sample_ids(path)
        ids.append(sample); params.append(param); reals.append(real)

    if not predictions:
        raise ValueError("No input file contained the required 25 snapshots.")

    y_pred = np.stack(predictions).astype(np.float32)
    y_true = np.stack(targets).astype(np.float32)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        y_pred=y_pred,
        y_true=y_true,
        sample_id=np.asarray(ids),
        param_id=np.asarray(params),
        real_id=np.asarray(reals),
        timestep_indices=np.asarray(FUTURE),
        observed_indices=np.asarray(OBSERVED),
        model_name=np.asarray(args.baseline),
        variant=np.asarray(args.baseline),
        seed=np.asarray(0),
        protocol=np.asarray("extrap"),
        transform_metadata=np.asarray(json.dumps({"space": "log10_C", "eps_c": EPS_C})),
    )
    print(f"Wrote {out} with shape {list(y_pred.shape)}")


if __name__ == "__main__":
    main()
