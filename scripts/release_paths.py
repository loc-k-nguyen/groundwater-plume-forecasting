"""Prevent reproduction commands from writing over released sources or evidence."""
from __future__ import annotations

from pathlib import Path

PROTECTED = {'.git', '.github', 'tables', 'metadata', 'configs', 'src',
             'scripts', 'docs', 'tests', 'figures'}


def safe_output_directory(root: Path, output: Path) -> Path:
    """Resolve symlinks and reject source directories, the root and its ancestors."""
    root, output = root.resolve(), output.resolve()
    if output == root or output in root.parents:
        raise ValueError('Output must not be the release root or one of its ancestors')
    for name in PROTECTED:
        protected = (root/name).resolve()
        if output == protected or protected in output.parents:
            raise ValueError('Refusing to overwrite released sources or evidence')
    return output
