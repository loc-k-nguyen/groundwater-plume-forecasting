"""
Sequence-aware patch datasets for temporal forecasting protocols.

Two protocols:
  extrap: train t=0..14, test future t=15..24
  interp: train even indices {0,2,...,24}, test odd indices {1,3,...,23}

Both protocols require the model to see the full K-field and a subset of the
concentration sequence during training, then predict at held-out timesteps.

This module provides:
  GroundwaterSequencePatchDataset: base class for both protocols
  ExtrapolationDataset: extrapolation protocol
  InterpolationDataset: interpolation protocol
  build_files(): helper shared with trainers
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset_timecond_patch import normalize_K, log10_C
from src.data.io import flatten_param_folders, load_param_split


# Alternating index sets for the interpolation protocol
INTERP_TRAIN_INDICES: List[int] = list(range(0, 25, 2))   # {0,2,4,...,24}, 13 timesteps
INTERP_TEST_INDICES:  List[int] = list(range(1, 25, 2))   # {1,3,5,...,23}, 12 timesteps

# Extrapolation index sets
EXTRAP_TRAIN_INDICES: List[int] = list(range(0, 15))       # t=0..14, 15 timesteps
EXTRAP_TEST_INDICES:  List[int] = list(range(15, 25))      # t=15..24, 10 timesteps


class GroundwaterSequencePatchDataset(Dataset):
    """
    Returns a random spatial patch for one file and provides the concentration
    sequence at a specified subset of timesteps.

    Each __getitem__ returns:
      K_patch       (1, P, P)         normalised log-conductivity
      C_train_log   (T_train, P, P)   log10(C) at train timestep indices
      C_test_log    (T_test, P, P)    log10(C) at test timestep indices
      C_train_raw   (T_train, P, P)   physical C at train timestep indices
      times_train   (T_train,)        normalised time values [0,1] for train set
      times_test    (T_test,)         normalised time values [0,1] for test set
      times_all     (T,)              all normalised time values

    The train/test split of timesteps is fixed at construction via
    train_indices and test_indices.
    """

    def __init__(
        self,
        files: List[str],
        stats: Dict,
        train_indices: List[int],
        test_indices: List[int],
        patch_size: int = 320,
        use_logK: bool = True,
        eps_k: float = 1e-6,
        eps_c: float = 1e-12,
        seed: int = 0,
    ):
        self.files = files
        self.stats = stats
        self.train_indices = sorted(int(i) for i in train_indices)
        self.test_indices  = sorted(int(i) for i in test_indices)
        self.patch_size    = int(patch_size)
        self.use_logK      = bool(use_logK)
        self.eps_k         = float(eps_k)
        self.eps_c         = float(eps_c)
        self.seed          = int(seed)

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + 1000003 * idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp = self.files[idx]
        rng = self._rng(idx)

        with np.load(fp) as d:
            K = d["K"].astype(np.float32)       # (H, W)
            C = d["C"].astype(np.float32)       # (T, H, W)
            times = d["times"].astype(np.float32)  # (T,)

        H, W = K.shape
        T = C.shape[0]
        P = self.patch_size

        if H < P or W < P:
            raise ValueError(f"Patch {P} too large for field {H}x{W} in {fp}")

        # random crop
        y0 = int(rng.integers(0, H - P + 1))
        x0 = int(rng.integers(0, W - P + 1))

        K_patch = K[y0:y0 + P, x0:x0 + P]
        C_patch = C[:, y0:y0 + P, x0:x0 + P]   # (T, P, P)

        # filter valid indices
        train_idx = [i for i in self.train_indices if i < T]
        test_idx  = [i for i in self.test_indices  if i < T]

        if not train_idx:
            raise ValueError(f"No valid train indices for T={T} in {fp}")

        # normalise time to [0, 1] over the full sequence
        t0, t1 = float(times.min()), float(times.max())
        t_range = max(t1 - t0, 1e-12)
        times_norm = ((times - t0) / t_range).astype(np.float32)

        # log10 concentration
        C_log = log10_C(C_patch, self.eps_c)   # (T, P, P)

        # K normalisation
        K_norm = normalize_K(K_patch, self.stats, self.use_logK, self.eps_k)

        return {
            "K_patch":     torch.from_numpy(K_norm[None, ...]),             # (1, P, P)
            "C_train_log": torch.from_numpy(C_log[train_idx]),              # (T_tr, P, P)
            "C_test_log":  torch.from_numpy(C_log[test_idx]),               # (T_te, P, P)
            "C_train_raw": torch.from_numpy(C_patch[train_idx]),            # (T_tr, P, P)
            "times_train": torch.from_numpy(times_norm[train_idx]),         # (T_tr,)
            "times_test":  torch.from_numpy(times_norm[test_idx]),          # (T_te,)
            "times_all":   torch.from_numpy(times_norm),                    # (T,)
            "train_indices": torch.tensor(train_idx, dtype=torch.int64),
            "test_indices":  torch.tensor(test_idx,  dtype=torch.int64),
        }


class ExtrapolationDataset(GroundwaterSequencePatchDataset):
    """
    Primary protocol: train t=0..14, test future t=15..24.
    Identical to the conference protocol.
    """

    def __init__(self, files: List[str], stats: Dict, **kwargs):
        super().__init__(
            files=files,
            stats=stats,
            train_indices=EXTRAP_TRAIN_INDICES,
            test_indices=EXTRAP_TEST_INDICES,
            **kwargs,
        )


class InterpolationDataset(GroundwaterSequencePatchDataset):
    """
    Secondary protocol: train on even timestep indices {0,2,...,24},
    test on odd indices {1,3,...,23}.
    """

    def __init__(self, files: List[str], stats: Dict, **kwargs):
        super().__init__(
            files=files,
            stats=stats,
            train_indices=INTERP_TRAIN_INDICES,
            test_indices=INTERP_TEST_INDICES,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _load_stats(stats_json_path: str) -> Dict:
    import json
    with open(stats_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["stats"] if isinstance(j, dict) and "stats" in j else j


def build_files(
    data_root: str,
    split_json: str,
    split: str,
) -> List[str]:
    """
    Return sorted list of .npz file paths for the given split.
    split: "train", "val", or "test"
    """
    train_folders, val_folders, test_folders = load_param_split(split_json)
    folder_map = {
        "train": train_folders,
        "val": val_folders,
        "test": test_folders,
    }
    if split not in folder_map:
        raise ValueError(f"Unknown split '{split}'. Expected one of train/val/test.")
    return flatten_param_folders(data_root, folder_map[split], file_glob="real_*.npz")


def build_dataset(
    protocol: str,
    split: str,
    data_root: str,
    split_json: str,
    stats: Dict,
    patch_size: int = 320,
    seed: int = 0,
) -> GroundwaterSequencePatchDataset:
    """
    Convenience factory.
    protocol: "extrap" or "interp"
    split:    "train", "val", or "test"
    """
    files = build_files(data_root, split_json, split)
    if not files:
        raise FileNotFoundError(
            f"No .npz files found for split={split} in {data_root}"
        )

    cls = ExtrapolationDataset if protocol == "extrap" else InterpolationDataset
    return cls(files=files, stats=stats, patch_size=patch_size, seed=seed)
