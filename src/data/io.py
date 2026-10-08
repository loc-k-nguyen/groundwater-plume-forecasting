import os
import json
import glob
from typing import List, Tuple, Dict


def load_param_split(split_json_path: str) -> Tuple[List[str], List[str], List[str]]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    S = j["splits"]
    return (
        S["train"]["param_folders"],
        S["val"]["param_folders"],
        S["test"]["param_folders"],
    )


def flatten_param_folders(data_root: str, param_folders: List[str], file_glob: str = "*.npz") -> List[str]:
    files = []
    for pf in param_folders:
        files.extend(sorted(glob.glob(os.path.join(data_root, pf, file_glob))))
    if not files:
        raise FileNotFoundError(
            f"No files found under data_root='{data_root}' for glob='{file_glob}'."
        )
    return files


def load_train_stats(stats_json_path: str) -> Dict[str, float]:
    with open(stats_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    # Stats files may be a flat mapping or wrap the values under a top-level
    # "stats" key; unwrap the latter so callers always get the flat mapping.
    if isinstance(j, dict) and "stats" in j:
        return j["stats"]
    return j