"""
Trainer for Transport-Aware Attentive Neural Process forecasting runs.

Primary protocol is extrapolation. The CLI keeps ablation flags so the full
method and stripped variants are traceable from their saved config.json files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Allow execution from the repository root without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_interp import (
    ExtrapolationDataset,
    InterpolationDataset,
    _load_stats,
    build_files,
)
from src.models.ta_anp import GroundwaterTAANP

logger = logging.getLogger(__name__)


@dataclass
class CFG:
    data_root: str = "data"
    split_json: str = "metadata/param_split_fixed.json"
    stats_json: str = "metadata/train_stats.json"
    out_root: str = "runs/ta_anp/seed0"
    run_name: str = "obj2_jnl_ta_anp_extrap_seed0"
    protocol: str = "extrap"
    resume: bool = False
    seed: int = 0
    patch_size: int = 320
    batch_size: int = 1
    grad_accum_steps: int = 16
    num_workers: int = 0
    amp: bool = True
    epochs: int = 150
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    hidden_channels: int = 32
    token_dim: int = 64
    latent_dim: int = 32
    time_embed_dim: int = 32
    num_heads: int = 4
    kl_beta: float = 1e-4
    use_residual: bool = True
    use_k: bool = True
    use_flow: bool = True
    k_fusion: str = "append"
    lambda_plume: float = 0.5
    lambda_centroid: float = 0.1
    lambda_mass: float = 0.05
    y_bg: float = -12.0
    y_cap: float = 2.0
    eps_c: float = 1e-12
    plume_thresh: float = 1e-8
    data_range: float = 14.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_limit_files: Optional[int] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_phys(y_log: torch.Tensor, eps_c: float) -> torch.Tensor:
    return torch.clamp(torch.pow(10.0, y_log.float()) - eps_c, min=0.0)


def masked_plume_l1(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    eps_c: float,
    plume_thresh: float,
) -> torch.Tensor:
    target_phys = _to_phys(target_log, eps_c)
    mask = (target_phys > plume_thresh).float()
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred_log.float() - target_log.float()) * mask).sum() / denom


def centroid_loss(pred_log: torch.Tensor, target_log: torch.Tensor, eps_c: float) -> torch.Tensor:
    pred_phys = _to_phys(pred_log, eps_c)
    target_phys = _to_phys(target_log, eps_c)
    batch, steps, height, width = pred_phys.shape
    y = torch.arange(height, device=pred_log.device, dtype=pred_phys.dtype)
    x = torch.arange(width, device=pred_log.device, dtype=pred_phys.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    def centroid(field: torch.Tensor) -> torch.Tensor:
        mass = field.sum(dim=(-2, -1)).clamp_min(1e-8)
        cx = (field * xx).sum(dim=(-2, -1)) / mass
        cy = (field * yy).sum(dim=(-2, -1)) / mass
        return torch.stack([cx, cy], dim=-1)

    err_px = torch.linalg.vector_norm(centroid(pred_phys) - centroid(target_phys), dim=-1)
    return err_px.mean() / float(max(height, width))


def integrated_concentration_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    eps_c: float,
) -> torch.Tensor:
    pred_mass = _to_phys(pred_log, eps_c).sum(dim=(-2, -1))
    target_mass = _to_phys(target_log, eps_c).sum(dim=(-2, -1)).clamp_min(1e-8)
    return (pred_mass - target_mass).abs().div(target_mass).mean()


def disable_auxiliary_losses(cfg: CFG) -> None:
    """Apply the archived no-auxiliary-loss ablation exactly."""
    cfg.lambda_plume = 0.0
    cfg.lambda_centroid = 0.0
    cfg.lambda_mass = 0.0


def _ssim_patch(pred: torch.Tensor, gt: torch.Tensor, data_range: float) -> float:
    pred = pred.float()
    gt = gt.float()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = pred.mean()
    mu_y = gt.mean()
    s_x = ((pred - mu_x) ** 2).mean()
    s_y = ((gt - mu_y) ** 2).mean()
    s_xy = ((pred - mu_x) * (gt - mu_y)).mean()
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * s_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (s_x + s_y + c2)
    return float((num / den).item())


def train_one_epoch(
    model: GroundwaterTAANP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: CFG,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "plume": 0.0, "centroid": 0.0, "mass": 0.0, "kl": 0.0}
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        k_field = batch["K_patch"].to(cfg.device, non_blocking=True)
        c_obs = batch["C_train_log"].to(cfg.device, non_blocking=True)
        c_target = batch["C_test_log"].to(cfg.device, non_blocking=True)
        t_obs = batch["times_train"].to(cfg.device, non_blocking=True)
        t_target = batch["times_test"].to(cfg.device, non_blocking=True)

        with autocast(enabled=cfg.amp):
            preds, aux = model(c_obs, k_field, t_obs, t_target, c_target=c_target, times_target=t_target)
            recon = F.l1_loss(preds.float(), c_target.float())
            plume = masked_plume_l1(preds, c_target, cfg.eps_c, cfg.plume_thresh)
            cent = centroid_loss(preds, c_target, cfg.eps_c)
            mass = integrated_concentration_loss(preds, c_target, cfg.eps_c)
            kl = model.kl_divergence(aux["post_mu"], aux["post_logvar"], aux["prior_mu"], aux["prior_logvar"])
            loss = (
                recon
                + cfg.lambda_plume * plume
                + cfg.lambda_centroid * cent
                + cfg.lambda_mass * mass
                + cfg.kl_beta * kl
            )

        scaler.scale(loss / max(cfg.grad_accum_steps, 1)).backward()
        should_step = (batch_idx + 1) % max(cfg.grad_accum_steps, 1) == 0 or (batch_idx + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(loss.item())
        totals["recon"] += float(recon.item())
        totals["plume"] += float(plume.item())
        totals["centroid"] += float(cent.item())
        totals["mass"] += float(mass.item())
        totals["kl"] += float(kl.item())
        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def evaluate(model: GroundwaterTAANP, files: List[str], stats: Dict, cfg: CFG, split_name: str = "test") -> Dict[str, float]:
    from src.data.dataset_interp import (
        EXTRAP_TEST_INDICES,
        EXTRAP_TRAIN_INDICES,
        INTERP_TEST_INDICES,
        INTERP_TRAIN_INDICES,
    )
    from src.data.dataset_timecond_patch import log10_C, normalize_K

    model.eval()
    if cfg.eval_limit_files:
        files = files[: cfg.eval_limit_files]

    obs_idx = EXTRAP_TRAIN_INDICES if cfg.protocol == "extrap" else INTERP_TRAIN_INDICES
    target_idx = EXTRAP_TEST_INDICES if cfg.protocol == "extrap" else INTERP_TEST_INDICES
    ssim_vals: List[float] = []

    for fp in files:
        with np.load(fp) as data:
            k_arr = data["K"].astype(np.float32)
            c_arr = data["C"].astype(np.float32)
            times = data["times"].astype(np.float32)

        valid_obs = [i for i in obs_idx if i < c_arr.shape[0]]
        valid_target = [i for i in target_idx if i < c_arr.shape[0]]
        if not valid_obs or not valid_target:
            continue

        t0 = float(times.min())
        t_range = max(float(times.max()) - t0, 1e-12)
        times_norm = ((times - t0) / t_range).astype(np.float32)
        k_norm = normalize_K(k_arr, stats, use_logK=True, eps_k=1e-6)
        c_log = log10_C(c_arr, eps_c=cfg.eps_c)

        k_t = torch.from_numpy(k_norm[None, None]).to(cfg.device)
        c_obs = torch.from_numpy(c_log[valid_obs][None]).to(cfg.device)
        c_target = torch.from_numpy(c_log[valid_target][None]).to(cfg.device)
        t_obs = torch.from_numpy(times_norm[valid_obs][None]).to(cfg.device)
        t_target = torch.from_numpy(times_norm[valid_target][None]).to(cfg.device)

        with autocast(enabled=cfg.amp):
            preds, _ = model(c_obs, k_t, t_obs, t_target, deterministic=True)

        for idx in range(preds.shape[1]):
            ssim_vals.append(
                _ssim_patch(
                    preds[0, idx : idx + 1].clamp(cfg.y_bg, cfg.y_cap),
                    c_target[0, idx : idx + 1],
                    cfg.data_range,
                )
            )

    return {"mean_ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0, f"{split_name}_n_files": len(files)}


def build_model(cfg: CFG) -> GroundwaterTAANP:
    return GroundwaterTAANP(
        hidden_channels=cfg.hidden_channels,
        token_dim=cfg.token_dim,
        latent_dim=cfg.latent_dim,
        time_embed_dim=cfg.time_embed_dim,
        num_heads=cfg.num_heads,
        kl_beta=cfg.kl_beta,
        use_residual=cfg.use_residual,
        use_k=cfg.use_k,
        use_flow=cfg.use_flow,
        k_fusion=cfg.k_fusion,
        y_bg=cfg.y_bg,
        y_cap=cfg.y_cap,
        eps_c=cfg.eps_c,
    )


def main(cfg: CFG) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(cfg.seed)

    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(cfg), handle, indent=2)

    stats = _load_stats(cfg.stats_json)
    dataset_cls = ExtrapolationDataset if cfg.protocol == "extrap" else InterpolationDataset
    train_files = build_files(cfg.data_root, cfg.split_json, "train")
    val_files = build_files(cfg.data_root, cfg.split_json, "val")
    test_files = build_files(cfg.data_root, cfg.split_json, "test")
    ds_train = dataset_cls(train_files, stats, patch_size=cfg.patch_size, seed=cfg.seed)
    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = build_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)

    best_val_ssim = -1.0
    start_epoch = 0
    latest_ckpt = os.path.join(out_dir, "checkpoint_latest.pt")
    best_ckpt = os.path.join(out_dir, "checkpoint_best.pt")
    if cfg.resume and os.path.isfile(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if cfg.amp and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_ssim = ckpt.get("best_val_ssim", -1.0)
        logger.info("Resuming %s from epoch %d", cfg.run_name, start_epoch)
    elif cfg.resume:
        logger.info("No checkpoint found for %s; starting fresh", cfg.run_name)

    logger.info("Starting %s | protocol=%s | seed=%d | train=%d | val=%d", cfg.run_name, cfg.protocol, cfg.seed, len(train_files), len(val_files))
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, loader_train, optimizer, scaler, cfg)
        scheduler.step()
        val_metrics = evaluate(model, val_files, stats, cfg, split_name="val")
        val_ssim = val_metrics["mean_ssim"]
        row = {
            "epoch": epoch,
            **train_metrics,
            "mean_ssim": val_ssim,
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_s": time.time() - t0,
        }
        with open(os.path.join(out_dir, "train_log.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        logger.info(
            "ep %03d | loss=%.4f recon=%.4f plume=%.4f cent=%.4f mass=%.4f kl=%.4f val_ssim=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["recon"],
            train_metrics["plume"],
            train_metrics["centroid"],
            train_metrics["mass"],
            train_metrics["kl"],
            val_ssim,
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if cfg.amp else None,
            "best_val_ssim": best_val_ssim,
            "cfg": vars(cfg),
        }
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            state["best_val_ssim"] = best_val_ssim
            torch.save(state, best_ckpt)
        torch.save(state, latest_ckpt)

    logger.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_files, stats, cfg, split_name="test")
    logger.info("Test SSIM: %.4f", test_metrics["mean_ssim"])
    results = {
        "run_name": cfg.run_name,
        "best_val_ssim": best_val_ssim,
        "test_ssim": test_metrics["mean_ssim"],
        "test_n_files": test_metrics.get("test_n_files", len(test_files)),
        "cfg": vars(cfg),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Done. Results saved to %s/results.json", out_dir)


def _parse() -> CFG:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=CFG.data_root)
    parser.add_argument("--split_json", default=CFG.split_json)
    parser.add_argument("--stats_json", default=CFG.stats_json)
    parser.add_argument("--out_root", default=CFG.out_root)
    parser.add_argument("--run_name", default=CFG.run_name)
    parser.add_argument("--protocol", default=CFG.protocol, choices=["extrap", "interp"])
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--patch_size", type=int, default=CFG.patch_size)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=CFG.grad_accum_steps)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--grad_clip_norm", type=float, default=CFG.grad_clip_norm)
    parser.add_argument("--hidden_channels", type=int, default=CFG.hidden_channels)
    parser.add_argument("--token_dim", type=int, default=CFG.token_dim)
    parser.add_argument("--latent_dim", type=int, default=CFG.latent_dim)
    parser.add_argument("--time_embed_dim", type=int, default=CFG.time_embed_dim)
    parser.add_argument("--num_heads", type=int, default=CFG.num_heads)
    parser.add_argument("--kl_beta", type=float, default=CFG.kl_beta)
    parser.add_argument("--no_residual", action="store_false", dest="use_residual")
    parser.add_argument("--no_k", action="store_false", dest="use_k")
    parser.add_argument("--no_flow", action="store_false", dest="use_flow")
    parser.add_argument("--k_fusion", default=CFG.k_fusion, choices=["append", "film"])
    parser.add_argument("--lambda_plume", type=float, default=CFG.lambda_plume)
    parser.add_argument("--lambda_centroid", type=float, default=CFG.lambda_centroid)
    parser.add_argument("--lambda_mass", type=float, default=CFG.lambda_mass)
    parser.add_argument(
        "--no_auxiliary_losses",
        "--no_transport_loss",
        dest="no_auxiliary_losses",
        action="store_true",
        help=(
            "disable plume, centroid, and integrated-concentration auxiliary losses; "
            "--no_transport_loss is retained as a backward-compatible alias"
        ),
    )
    parser.add_argument("--amp", action="store_true", default=CFG.amp)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--eval_limit_files", type=int, default=None)
    args = parser.parse_args()

    cfg = CFG()
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if args.no_auxiliary_losses:
        disable_auxiliary_losses(cfg)
    return cfg


if __name__ == "__main__":
    main(_parse())
