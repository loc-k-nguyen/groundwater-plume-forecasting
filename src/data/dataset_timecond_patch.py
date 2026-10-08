import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_stats(stats_json_path: str) -> Dict:
    import json
    with open(stats_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["stats"] if isinstance(j, dict) and "stats" in j else j


def normalize_K(K: np.ndarray, stats: Dict, use_logK: bool, eps_k: float) -> np.ndarray:
    K = K.astype(np.float64)
    if use_logK:
        K = np.log(np.clip(K, eps_k, None))
    K = (K - float(stats["k_mean"])) / (float(stats["k_std"]) + 1e-8)
    return K.astype(np.float32)


def log10_C(C: np.ndarray, eps_c: float) -> np.ndarray:
    # C is physical (>=0), return log10(C + eps)
    C = np.clip(C.astype(np.float32), 0.0, None)
    return np.log10(C + eps_c).astype(np.float32)


class GroundwaterTimeCondPatchDatasetLogC(Dataset):
    """
    Time-conditioned patch dataset:
      input  : (2, P, P) = [K_norm, t_channel]
      target : (1, P, P) = log10(C(t) + eps_c)
      also returns C_raw for weighting/mass penalty.

    Each sample: choose a file (param,real), choose a timestep index t_idx,
                 then random crop patch (P,P).
    """

    def __init__(
        self,
        files: List[str],
        stats: Dict,
        patch_size: int = 256,
        use_logK: bool = True,
        eps_k: float = 1e-6,
        eps_c: float = 1e-12,
        seed: int = 0,
        t_index_mode: str = "uniform",   # "uniform" or "mid_bias"
        mid_bias_sigma: float = 0.18,    # for "mid_bias"
        allowed_timestep_indices: Optional[List[int]] = None,
        time_encoding: str = "linear",
        use_context_cond: bool = False,
    ):
        self.files = files
        self.stats = stats
        self.patch = int(patch_size)
        self.use_logK = bool(use_logK)
        self.eps_k = float(eps_k)
        self.eps_c = float(eps_c)
        self.seed = int(seed)

        self.t_index_mode = str(t_index_mode)
        self.mid_bias_sigma = float(mid_bias_sigma)
        self.allowed_timestep_indices = (
            sorted(int(t) for t in allowed_timestep_indices)
            if allowed_timestep_indices is not None
            else None
        )
        self.time_encoding = str(time_encoding)
        self.use_context_cond = bool(use_context_cond)

        # cache minimal metadata for speed (H,W,T, tmin,tmax)
        self._meta = {}
        for fp in self.files[: min(len(self.files), 200)]:  # light warm-up
            try:
                with np.load(fp) as d:
                    C = d["C"]
                    times = d["times"]
                self._meta[fp] = (C.shape[0], C.shape[1], C.shape[2], float(times.min()), float(times.max()))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.files)

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        # deterministic per index (works nicely even with num_workers=0)
        return np.random.default_rng(self.seed + 1000003 * idx)

    def _valid_timestep_indices(self, T: int) -> np.ndarray:
        if self.allowed_timestep_indices is None:
            return np.arange(T, dtype=np.int64)

        valid = np.array(
            [t for t in self.allowed_timestep_indices if 0 <= int(t) < T],
            dtype=np.int64,
        )
        if valid.size == 0:
            raise ValueError(
                f"No valid allowed_timestep_indices remain for T={T}: {self.allowed_timestep_indices}"
            )
        return valid

    def _choose_timestep(self, rng: np.random.Generator, T: int) -> int:
        valid = self._valid_timestep_indices(T)

        if self.t_index_mode == "uniform":
            return int(rng.choice(valid))

        if self.t_index_mode == "mid_bias":
            # sample around middle of timeline (helps transition regime)
            # build discrete probabilities once per call (T=25 so fine)
            x = np.linspace(0.0, 1.0, T, dtype=np.float64)
            mu = 0.5
            sig = max(self.mid_bias_sigma, 1e-6)
            p = np.exp(-0.5 * ((x - mu) / sig) ** 2)
            p = p[valid]
            p = p / p.sum()
            return int(rng.choice(valid, p=p))

        raise ValueError(f"Unknown t_index_mode={self.t_index_mode}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp = self.files[idx]
        rng = self._rng_for_index(idx)

        with np.load(fp) as d:
            K = d["K"].astype(np.float32)       # (H,W)
            C = d["C"].astype(np.float32)       # (T,H,W)
            times = d["times"].astype(np.float32)  # (T,)

        H, W = K.shape
        T = C.shape[0]

        # timestep
        t_idx = self._choose_timestep(rng, T)
        t = float(times[t_idx])
        t0 = float(times.min())
        t1 = float(times.max())
        t_norm = (t - t0) / (t1 - t0 + 1e-12)  # [0,1]

        # random crop
        P = self.patch
        if H < P or W < P:
            raise ValueError(f"Patch {P} too big for field {H}x{W} in {fp}")

        y0 = int(rng.integers(0, H - P + 1))
        x0 = int(rng.integers(0, W - P + 1))

        K_patch = K[y0:y0 + P, x0:x0 + P]
        C_raw = C[t_idx, y0:y0 + P, x0:x0 + P]
        seen_log_seq = log10_C(C[:15, y0:y0 + P, x0:x0 + P], self.eps_c)
        c_ctx_patch = None
        if self.use_context_cond:
            if T <= 14:
                raise ValueError(f"Context timestep index 14 is unavailable for T={T} in {fp}")
            c_ctx = np.log10(np.clip(C[14], 0.0, None) + self.eps_c).astype(np.float32)
            c_ctx_patch = c_ctx[y0:y0 + P, x0:x0 + P]

        K_norm = normalize_K(K_patch, self.stats, self.use_logK, self.eps_k)
        Y_log = log10_C(C_raw, self.eps_c)

        if self.time_encoding == "linear":
            t_chan = np.full((P, P), t_norm, dtype=np.float32)
            chans = [K_norm, t_chan]
        elif self.time_encoding == "fourier":
            twopi = 2.0 * np.pi
            sin_t = np.full((P, P), np.sin(twopi * t_norm), dtype=np.float32)
            cos_t = np.full((P, P), np.cos(twopi * t_norm), dtype=np.float32)
            chans = [K_norm, sin_t, cos_t]
        else:
            raise ValueError(f"Unknown time_encoding={self.time_encoding}")
        if c_ctx_patch is not None:
            chans.append(c_ctx_patch)
        X = np.stack(chans, axis=0)

        # torch
        # X = torch.from_numpy(X)  # (3,P,P)
        Y = torch.from_numpy(Y_log[None, ...])           # (1,P,P)
        C_raw_t = torch.from_numpy(C_raw[None, ...])     # (1,P,P)

        return {
            "X": X,
            "Y_log": Y,
            "C_raw": C_raw_t,
            "seen_log_seq": torch.from_numpy(seen_log_seq),
            "t_norm": torch.tensor([t_norm], dtype=torch.float32),
            "t_idx": torch.tensor([t_idx], dtype=torch.int64),
        }
