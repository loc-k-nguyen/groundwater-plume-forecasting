"""Build a SHA-256 release manifest for the repository tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "outputs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Verify the saved manifest without writing it")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = (args.out or root / "MANIFEST.csv").resolve()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() != out
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"path": path.relative_to(root).as_posix(), "bytes": str(path.stat().st_size), "sha256": sha256(path)}
            for path in sorted(files, key=lambda item: item.relative_to(root).as_posix())]
    if args.check:
        with out.open(newline="", encoding="utf-8") as handle:
            saved = list(csv.DictReader(handle))
        if saved != rows:
            raise ValueError("Manifest file set, sizes, ordering or hashes do not match")
        print(f"PASS: manifest verifies {len(rows)} files")
        return
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} with {len(files)} entries")


if __name__ == "__main__":
    main()
