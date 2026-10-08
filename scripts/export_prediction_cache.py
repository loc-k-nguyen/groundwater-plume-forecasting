"""Export TA-ANP future prediction caches.

Minimal cache writer for trained TA-ANP checkpoints. It uses the shared
model-loading and inference path and
writes predictions and targets to NPZ for downstream shape metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.evaluate_query_model import (
    build_model,
    device_or_cpu,
    protocol_indices,
    query_model,
)
from src.data.dataset_timecond_patch import log10_C, normalize_K
from src.data.io import flatten_param_folders, load_param_split, load_train_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TA-ANP prediction cache.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--stats_json", required=True)
    parser.add_argument("--protocol", choices=["extrap", "interp"], default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_limit_files", type=int, default=0)
    parser.add_argument("--model_name", required=True, help="e.g. TA-ANP-noK")
    parser.add_argument("--variant", required=True, help="e.g. ta_anp_no_k")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache_out_npz", required=True)
    parser.add_argument("--meta_out_json", default=None)
    parser.add_argument(
        "--include_seen",
        action="store_true",
        help="Include observed history in the NPZ. Omit for smaller full caches.",
    )
    return parser.parse_args()


def infer_ids(path: str) -> tuple[str, str, str]:
    normalized = Path(path).as_posix()
    param_match = re.search(r"(param_[^/]+)", normalized)
    real_match = re.search(r"(real_[^/.]+)", normalized)
    param_id = param_match.group(1) if param_match else ""
    real_id = real_match.group(1) if real_match else Path(path).stem
    return f"{param_id}/{real_id}", param_id, real_id


def validate_arrays(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {y_pred.shape} vs {y_true.shape}")
    if y_pred.ndim != 4:
        raise ValueError(f"Expected y_pred shape (N,T,H,W), got {y_pred.shape}")
    if np.isnan(y_pred).any():
        raise ValueError("NaN detected in y_pred")
    if np.isnan(y_true).any():
        raise ValueError("NaN detected in y_true")
    return {
        "y_pred_shape": list(y_pred.shape),
        "y_true_shape": list(y_true.shape),
        "n_samples": int(y_pred.shape[0]),
        "n_timesteps": int(y_pred.shape[1]),
        "y_pred_min": float(np.min(y_pred)),
        "y_pred_max": float(np.max(y_pred)),
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
        "y_pred_has_nan": bool(np.isnan(y_pred).any()),
        "y_true_has_nan": bool(np.isnan(y_true).any()),
    }


def discover_test_files(data_root: str, test_params: list[str]) -> list[str]:
    try:
        return flatten_param_folders(data_root, test_params)
    except FileNotFoundError:
        root = Path(data_root)
        files: list[str] = []
        for param in test_params:
            files.extend(str(p) for p in sorted(root.glob(f"*/{param}/*.npz")))
            files.extend(str(p) for p in sorted(root.glob(f"{param}/real_*.npz")))
        if not files:
            raise
        return files


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = device_or_cpu(args.device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)
    if "model" not in ckpt:
        raise KeyError("Checkpoint does not contain a 'model' state_dict")
    run_cfg = dict(ckpt.get("cfg", {}))
    protocol = args.protocol or str(run_cfg.get("protocol", "extrap"))
    obs_idx, query_idx = protocol_indices(protocol)

    _, _, test_params = load_param_split(args.split_json)
    test_files = discover_test_files(args.data_root, test_params)
    if args.eval_limit_files > 0:
        test_files = test_files[: args.eval_limit_files]
    if not test_files:
        raise RuntimeError("No test files found for cache export")

    stats = load_train_stats(args.stats_json)
    model = build_model("ta_anp", run_cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    use_amp = bool(run_cfg.get("amp", True)) and device.startswith("cuda")
    eps_c = float(run_cfg.get("eps_c", 1e-12))

    y_pred_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    seen_list: list[np.ndarray] = []
    sample_ids: list[str] = []
    param_ids: list[str] = []
    real_ids: list[str] = []

    for file_path in test_files:
        with np.load(file_path) as data:
            k_field = data["K"].astype(np.float32)
            c_field = data["C"].astype(np.float32)
            times = data["times"].astype(np.float32)

        total_t = int(c_field.shape[0])
        valid_obs = [int(i) for i in obs_idx if int(i) < total_t]
        valid_query = [int(i) for i in query_idx if int(i) < total_t]
        if len(valid_obs) != len(obs_idx) or len(valid_query) != len(query_idx):
            continue

        k_norm = normalize_K(k_field, stats, use_logK=True, eps_k=1e-6)
        c_log = log10_C(c_field, eps_c=eps_c)
        t0 = float(times.min())
        t_range = max(float(times.max()) - t0, 1e-12)
        times_norm = ((times - t0) / t_range).astype(np.float32)

        k_t = torch.from_numpy(k_norm[None, None]).to(device)
        c_obs = torch.from_numpy(c_log[valid_obs][None]).to(device)
        t_obs = torch.from_numpy(times_norm[valid_obs][None]).to(device)
        t_query = torch.from_numpy(times_norm[valid_query][None]).to(device)

        preds = query_model(model, "ta_anp", c_obs, k_t, t_obs, t_query, use_amp)
        pred_log = preds[0].detach().float().cpu().numpy().astype(np.float32)
        true_log = c_log[valid_query].astype(np.float32)
        seen_log = c_log[valid_obs].astype(np.float32)

        sample_id, param_id, real_id = infer_ids(file_path)
        y_pred_list.append(pred_log)
        y_true_list.append(true_log)
        seen_list.append(seen_log)
        sample_ids.append(sample_id)
        param_ids.append(param_id)
        real_ids.append(real_id)

    if not y_pred_list:
        raise RuntimeError("No valid samples exported")

    y_pred = np.stack(y_pred_list, axis=0).astype(np.float32)
    y_true = np.stack(y_true_list, axis=0).astype(np.float32)
    seen_log_c = np.stack(seen_list, axis=0).astype(np.float32) if args.include_seen else None
    validation = validate_arrays(y_pred, y_true)

    out_npz = Path(args.cache_out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "y_pred": y_pred,
        "y_true": y_true,
        "sample_id": np.asarray(sample_ids),
        "param_id": np.asarray(param_ids),
        "real_id": np.asarray(real_ids),
        "timestep_indices": np.asarray(query_idx, dtype=np.int64),
        "observed_indices": np.asarray(obs_idx, dtype=np.int64),
        "model_name": np.asarray(args.model_name),
        "variant": np.asarray(args.variant),
        "seed": np.asarray(args.seed, dtype=np.int64),
        "protocol": np.asarray(protocol),
        "transform_metadata": np.asarray(
            json.dumps(
                {
                    "space": "log10_C",
                    "eps_c": eps_c,
                    "y_bg": float(run_cfg.get("y_bg", -12.0)),
                    "y_cap": float(run_cfg.get("y_cap", 2.0)),
                    "data_range": float(run_cfg.get("data_range", 14.0)),
                    "split_json": args.split_json,
                    "stats_json": args.stats_json,
                },
                sort_keys=True,
            )
        ),
    }
    if seen_log_c is not None:
        payload["seen_log_c"] = seen_log_c
        validation["seen_log_c_shape"] = list(seen_log_c.shape)
    np.savez_compressed(out_npz, **payload)

    meta = {
        "model_name": args.model_name,
        "variant": args.variant,
        "seed": args.seed,
        "protocol": protocol,
        "checkpoint": str(checkpoint),
        "cache_out_npz": str(out_npz),
        "device_requested": args.device,
        "device_used": device,
        "eval_limit_files": int(args.eval_limit_files),
        "include_seen": bool(args.include_seen),
        "timestep_indices": list(query_idx),
        "observed_indices": list(obs_idx),
        "run_cfg": run_cfg,
        "validation": validation,
    }
    meta_path = Path(args.meta_out_json) if args.meta_out_json else out_npz.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print("[export_ta_anp_prediction_cache] wrote cache")
    print(f"  npz: {out_npz}")
    print(f"  meta: {meta_path}")
    print(f"  y_pred_shape: {validation['y_pred_shape']}")
    print(f"  y_true_shape: {validation['y_true_shape']}")
    print(f"  y_pred_range: {validation['y_pred_min']:.6f} .. {validation['y_pred_max']:.6f}")
    print(f"  y_true_range: {validation['y_true_min']:.6f} .. {validation['y_true_max']:.6f}")


if __name__ == "__main__":
    main()
