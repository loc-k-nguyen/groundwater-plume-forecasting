"""Audit included source, sensitive file classes and immutable artifact records."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', 'outputs', '.venv', 'venv', '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache'}
FORBIDDEN = {'.npz', '.npy', '.pt', '.pth', '.ckpt', '.exe', '.dll', '.zip', '.eml', '.slurm', '.bat', '.sh'}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """Keep certification checks active under python -O as well as normal Python."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts)]
    parsed = 0
    for path in files:
        require(path.suffix.lower() not in FORBIDDEN, f'Excluded file class: {path.relative_to(ROOT)}')
        require(path.stat().st_size < 2_000_000, f'Oversized release file: {path.relative_to(ROOT)}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            parsed += 1
        if path.suffix in {'.py', '.json', '.csv', '.md', '.txt', '.cff', '.yml', '.yaml', '.svg', '.toml'}:
            content = path.read_text(encoding='utf-8-sig')
            require(not re.search(r'(?:[A-Z]:[\\/](?:Users|Khanh_Loc)|/(?:nfs|local/scratch|home)/)', content), f'Private path: {path.relative_to(ROOT)}')
            require(not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)', content), f'Credential pattern: {path.relative_to(ROOT)}')
            if path.suffix == '.json':
                require('MODFLOW2000 Layer Property Flow' not in content, f'Simulator deck: {path.relative_to(ROOT)}')
                require(not re.search(r'OPEN/CLOSE\s+[\"\x27]', content), f'Simulator file directive: {path.relative_to(ROOT)}')
                json.loads(content)
    cff = yaml.safe_load((ROOT/'CITATION.cff').read_text())
    require(cff['type'] == 'software' and cff['preferred-citation']['type'] == 'unpublished', 'Unverified manuscript citation status')
    require('doi' not in cff and 'doi' not in cff['preferred-citation'], 'Do not invent a publication DOI')
    manuscript = json.loads((ROOT/'docs/manuscript_snapshot.json').read_text())
    require(cff.get('version') == manuscript['final_release_tag'], 'Software version and manuscript mapping disagree')
    snapshot = json.loads((ROOT/'docs/source_snapshot.json').read_text())
    preparation_edits = {'scripts/build_manifest.py'}
    for row in snapshot:
        path = ROOT/row['path']
        require(path.is_file(), f'Missing source snapshot file: {row["path"]}')
        if row['path'] not in preparation_edits:
            require(digest(path) == row['copied_sha256'], f'Source snapshot hash mismatch: {row["path"]}')
    with (ROOT/'ARTIFACT_MAP.csv').open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            require(digest(ROOT/row['path']) == row['sha256'], f'Evidence hash mismatch: {row["path"]}')
    split = json.loads((ROOT/'metadata/param_split_fixed.json').read_text())
    groups = [set(split['splits'][key]['param_ids']) for key in ['train', 'val', 'test']]
    require(all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i+1, 3)), 'Overlapping configuration IDs')
    require(sum(map(len, groups)) == 162, 'Unexpected fixed-split configuration count')
    config = json.loads((ROOT/'configs/ta_anp_no_k.json').read_text())
    require(config['use_k'] is False and config['use_flow'] is True, 'Changed noK input regime')
    require(config['epochs'] == 150 and config['batch_size'] * config['grad_accum_steps'] == 16, 'Changed archived training protocol')
    print(f'PASS: {parsed} Python sources, {len(files)} files, snapshot hashes, fixed split, noK protocol, citation and content scan')

if __name__ == '__main__':
    main()
