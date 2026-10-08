"""
Evaluate query-based sequence models with plume-aware metrics.

Supported models:
  - EarthformerLite
  - NeuralProcess
  - TA-ANP

This module provides plume-aware evaluation reporting while
keeping the evaluation self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.cuda.amp import autocast

from src.models.earthformer_lite import GroundwaterEarthformerLite
from src.models.neural_process import GroundwaterNeuralProcess
from src.models.ta_anp import GroundwaterTAANP
from src.data.io import flatten_param_folders, load_param_split, load_train_stats
from src.eval.centroid_error import compute_plume_centroid_error
from src.eval.plume_ssim import compute_global_ssim, compute_plume_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate query models with plume-aware metrics.")
    parser.add_argument("--model_type", required=True, choices=["earthformerlite", "neuralprocess", "ta_anp"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--stats_json", required=True)
    parser.add_argument("--protocol", choices=["extrap", "interp"], default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seen_out_json", required=True)
    parser.add_argument("--future_out_json", required=True)
    parser.add_argument("--eval_limit_files", type=int, default=0)
    return parser.parse_args()


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def device_or_cpu(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


def protocol_indices(protocol: str) -> tuple[List[int], List[int]]:
    if protocol == "extrap":
        return list(range(0, 15)), list(range(15, 25))
    if protocol == "interp":
        return list(range(0, 25, 2)), list(range(1, 25, 2))
    raise ValueError(f"Unsupported protocol={protocol}")


def build_model(model_type: str, run_cfg: Dict, device: str) -> torch.nn.Module:
    if model_type == "earthformerlite":
        model = GroundwaterEarthformerLite(
            image_size=int(run_cfg.get("patch_size", 320)),
            patch_size=int(run_cfg.get("model_patch_size", 40)),
            embed_dim=int(run_cfg.get("embed_dim", 128)),
            depth=int(run_cfg.get("depth", 4)),
            num_heads=int(run_cfg.get("num_heads", 4)),
            mlp_ratio=float(run_cfg.get("mlp_ratio", 4.0)),
            dropout=float(run_cfg.get("dropout", 0.0)),
            y_bg=float(run_cfg.get("y_bg", -12.0)),
            y_cap=float(run_cfg.get("y_cap", 2.0)),
        ).to(device)
        return model

    if model_type == "neuralprocess":
        model = GroundwaterNeuralProcess(
            hidden_channels=int(run_cfg.get("hidden_channels", 32)),
            token_dim=int(run_cfg.get("token_dim", 64)),
            latent_dim=int(run_cfg.get("latent_dim", 32)),
            time_embed_dim=int(run_cfg.get("time_embed_dim", 32)),
            num_heads=int(run_cfg.get("num_heads", 4)),
            kl_beta=float(run_cfg.get("kl_beta", 1e-4)),
            y_bg=float(run_cfg.get("y_bg", -12.0)),
            y_cap=float(run_cfg.get("y_cap", 2.0)),
        ).to(device)
        return model

    if model_type == "ta_anp":
        model = GroundwaterTAANP(
            hidden_channels=int(run_cfg.get("hidden_channels", 32)),
            token_dim=int(run_cfg.get("token_dim", 64)),
            latent_dim=int(run_cfg.get("latent_dim", 32)),
            time_embed_dim=int(run_cfg.get("time_embed_dim", 32)),
            num_heads=int(run_cfg.get("num_heads", 4)),
            kl_beta=float(run_cfg.get("kl_beta", 1e-4)),
            use_residual=bool(run_cfg.get("use_residual", True)),
            use_k=bool(run_cfg.get("use_k", True)),
            use_flow=bool(run_cfg.get("use_flow", True)),
            k_fusion=str(run_cfg.get("k_fusion", "append")),
            y_bg=float(run_cfg.get("y_bg", -12.0)),
            y_cap=float(run_cfg.get("y_cap", 2.0)),
            eps_c=float(run_cfg.get("eps_c", 1e-12)),
        ).to(device)
        return model

    raise ValueError(f"Unsupported model_type={model_type}")


def query_model(
    model: torch.nn.Module,
    model_type: str,
    c_obs: torch.Tensor,
    k_field: torch.Tensor,
    times_obs: torch.Tensor,
    times_query: torch.Tensor,
    amp: bool,
) -> torch.Tensor:
    with autocast(enabled=amp):
        if model_type == "earthformerlite":
            return model(c_obs, k_field, times_obs, times_query)
        if model_type == "neuralprocess":
            preds, _ = model(c_obs, k_field, times_obs, times_query, deterministic=True)
            return preds
        if model_type == "ta_anp":
            preds, _ = model(c_obs, k_field, times_obs, times_query, deterministic=True)
            return preds
    raise ValueError(f"Unsupported model_type={model_type}")


@torch.no_grad()
def evaluate_indices(
    model: torch.nn.Module,
    model_type: str,
    files: Sequence[str],
    query_indices: Sequence[int],
    observed_indices: Sequence[int],
    stats: Dict,
    device: str,
    eps_c: float,
    use_amp: bool,
) -> Dict:
    from src.data.dataset_timecond_patch import log10_C, normalize_K

    all_ssim: List[float] = []
    all_plume_ssim: List[float] = []
    all_centroid_err: List[float] = []
    all_l1: List[float] = []
    all_mse: List[float] = []
    all_mass: List[float] = []
    ssim_per_t = None
    plume_ssim_per_t = None
    plume_valid_count_per_t = None
    centroid_err_per_t = None
    centroid_valid_count_per_t = None
    l1_per_t = None
    mse_per_t = None
    mass_per_t = None
    count_per_t = None

    model.eval()
    processed_files = 0

    for fp in files:
        with np.load(fp) as data:
            k_field = data["K"].astype(np.float32)
            c_field = data["C"].astype(np.float32)
            times = data["times"].astype(np.float32)

        total_t = int(c_field.shape[0])
        valid_obs = [int(i) for i in observed_indices if int(i) < total_t]
        valid_query = [int(i) for i in query_indices if int(i) < total_t]
        if not valid_obs or not valid_query:
            continue
        processed_files += 1

        k_norm = normalize_K(k_field, stats, use_logK=True, eps_k=1e-6)
        c_log = log10_C(c_field, eps_c=eps_c)
        t0 = float(times.min())
        t_range = max(float(times.max()) - t0, 1e-12)
        times_norm = ((times - t0) / t_range).astype(np.float32)

        k_t = torch.from_numpy(k_norm[None, None]).to(device)
        c_obs = torch.from_numpy(c_log[valid_obs][None]).to(device)
        t_obs = torch.from_numpy(times_norm[valid_obs][None]).to(device)
        t_query = torch.from_numpy(times_norm[valid_query][None]).to(device)

        preds = query_model(model, model_type, c_obs, k_t, t_obs, t_query, use_amp)
        pred_log = preds[0].detach().float().cpu().numpy()

        if ssim_per_t is None:
            ssim_per_t = np.zeros(total_t, dtype=np.float64)
            plume_ssim_per_t = np.zeros(total_t, dtype=np.float64)
            plume_valid_count_per_t = np.zeros(total_t, dtype=np.int64)
            centroid_err_per_t = np.zeros(total_t, dtype=np.float64)
            centroid_valid_count_per_t = np.zeros(total_t, dtype=np.int64)
            l1_per_t = np.zeros(total_t, dtype=np.float64)
            mse_per_t = np.zeros(total_t, dtype=np.float64)
            mass_per_t = np.zeros(total_t, dtype=np.float64)
            count_per_t = np.zeros(total_t, dtype=np.int64)

        for local_idx, t_idx in enumerate(valid_query):
            gt_phys = np.clip(c_field[t_idx], 0.0, None).astype(np.float32)
            gt_log = c_log[t_idx]
            pr_log = pred_log[local_idx].astype(np.float32)

            score = compute_global_ssim(gt_log, pr_log)
            plume_score, _ = compute_plume_ssim(gt_phys=gt_phys, pred_log=pr_log, eps_c=eps_c)
            pr_phys = np.clip((10.0 ** pr_log) - eps_c, 0.0, None).astype(np.float32)
            centroid_err, _ = compute_plume_centroid_error(gt_phys=gt_phys, pred_phys=pr_phys)
            diff = pr_phys - gt_phys
            l1_value = float(np.mean(np.abs(diff)))
            mse_value = float(np.mean(diff * diff))
            mass_value = float(np.abs(pr_phys.sum() - gt_phys.sum()))

            all_ssim.append(float(score))
            all_l1.append(l1_value)
            all_mse.append(mse_value)
            all_mass.append(mass_value)
            ssim_per_t[t_idx] += float(score)
            l1_per_t[t_idx] += l1_value
            mse_per_t[t_idx] += mse_value
            mass_per_t[t_idx] += mass_value
            count_per_t[t_idx] += 1

            if not np.isnan(plume_score):
                all_plume_ssim.append(float(plume_score))
                plume_ssim_per_t[t_idx] += float(plume_score)
                plume_valid_count_per_t[t_idx] += 1
            if not np.isnan(centroid_err):
                all_centroid_err.append(float(centroid_err))
                centroid_err_per_t[t_idx] += float(centroid_err)
                centroid_valid_count_per_t[t_idx] += 1

    if ssim_per_t is None:
        return {
            "mean_ssim_log": float("nan"),
            "mean_plume_ssim_log": float("nan"),
            "mean_centroid_err_px": float("nan"),
            "mean_l1_phys": float("nan"),
            "mean_mse_phys": float("nan"),
            "mean_mass_err_phys": float("nan"),
            "n_test_files": 0,
            "timestep_indices": list(query_indices),
        }

    evaluated = np.asarray([int(i) for i in query_indices], dtype=np.int64)
    return {
        "mean_ssim_log": float(np.mean(all_ssim)) if all_ssim else float("nan"),
        "mean_plume_ssim_log": float(np.mean(all_plume_ssim)) if all_plume_ssim else float("nan"),
        "mean_centroid_err_px": float(np.mean(all_centroid_err)) if all_centroid_err else float("nan"),
        "ssim_per_timestep_log": (ssim_per_t[evaluated] / np.maximum(count_per_t[evaluated], 1)).tolist(),
        "plume_ssim_per_timestep_log": (
            plume_ssim_per_t[evaluated] / np.maximum(plume_valid_count_per_t[evaluated], 1)
        ).tolist(),
        "plume_valid_count_per_timestep": plume_valid_count_per_t[evaluated].tolist(),
        "centroid_err_per_timestep_px": (
            centroid_err_per_t[evaluated] / np.maximum(centroid_valid_count_per_t[evaluated], 1)
        ).tolist(),
        "centroid_valid_count_per_timestep": centroid_valid_count_per_t[evaluated].tolist(),
        "mean_l1_phys": float(np.mean(all_l1)) if all_l1 else float("nan"),
        "l1_per_timestep_phys": (l1_per_t[evaluated] / np.maximum(count_per_t[evaluated], 1)).tolist(),
        "mean_mse_phys": float(np.mean(all_mse)) if all_mse else float("nan"),
        "mse_per_timestep_phys": (mse_per_t[evaluated] / np.maximum(count_per_t[evaluated], 1)).tolist(),
        "mean_mass_err_phys": float(np.mean(all_mass)) if all_mass else float("nan"),
        "mass_err_per_timestep_phys": (mass_per_t[evaluated] / np.maximum(count_per_t[evaluated], 1)).tolist(),
        "mean_summed_conc_abs_error": float(np.mean(all_mass)) if all_mass else float("nan"),
        "summed_conc_abs_error_per_timestep": (
            mass_per_t[evaluated] / np.maximum(count_per_t[evaluated], 1)
        ).tolist(),
        "n_test_files": processed_files,
        "timestep_indices": evaluated.tolist(),
        "eps_c": float(eps_c),
    }


def main() -> None:
    args = parse_args()
    device = device_or_cpu(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    run_cfg = dict(ckpt.get("cfg", {}))
    protocol = args.protocol or str(run_cfg.get("protocol", "extrap"))
    obs_idx, query_idx = protocol_indices(protocol)
    _, _, test_params = load_param_split(args.split_json)
    test_files = flatten_param_folders(args.data_root, test_params)
    if args.eval_limit_files > 0:
        test_files = test_files[: args.eval_limit_files]
    stats = load_train_stats(args.stats_json)

    model = build_model(args.model_type, run_cfg, device)
    model.load_state_dict(ckpt["model"])
    use_amp = bool(run_cfg.get("amp", True)) and device.startswith("cuda")
    eps_c = float(run_cfg.get("eps_c", 1e-12))

    metrics_seen = evaluate_indices(
        model=model,
        model_type=args.model_type,
        files=test_files,
        query_indices=obs_idx,
        observed_indices=obs_idx,
        stats=stats,
        device=device,
        eps_c=eps_c,
        use_amp=use_amp,
    )
    metrics_future = evaluate_indices(
        model=model,
        model_type=args.model_type,
        files=test_files,
        query_indices=query_idx,
        observed_indices=obs_idx,
        stats=stats,
        device=device,
        eps_c=eps_c,
        use_amp=use_amp,
    )

    save_json(args.seen_out_json, metrics_seen)
    save_json(args.future_out_json, metrics_future)

    print(f"[SeenOut] {args.seen_out_json}")
    print(f"[FutureOut] {args.future_out_json}")
    print(f"[SeenGlobalSSIM] {metrics_seen['mean_ssim_log']:.6f}")
    print(f"[SeenPlumeSSIM] {metrics_seen['mean_plume_ssim_log']:.6f}")
    print(f"[SeenCentroidErr] {metrics_seen['mean_centroid_err_px']:.3f}")
    print(f"[FutureGlobalSSIM] {metrics_future['mean_ssim_log']:.6f}")
    print(f"[FuturePlumeSSIM] {metrics_future['mean_plume_ssim_log']:.6f}")
    print(f"[FutureCentroidErr] {metrics_future['mean_centroid_err_px']:.3f}")


if __name__ == "__main__":
    main()
