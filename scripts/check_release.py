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
SKIP = {'.git', 'outputs', '.venv', 'venv', '__pycache__', '.pytest_cache'}
FORBIDDEN = {'.npz', '.npy', '.pt', '.pth', '.ckpt', '.exe', '.dll', '.zip', '.eml', '.slurm', '.bat', '.sh'}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts)]
    parsed = 0
    for path in files:
        assert path.suffix.lower() not in FORBIDDEN, path.relative_to(ROOT)
        assert path.stat().st_size < 2_000_000, path.relative_to(ROOT)
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            parsed += 1
        if path.suffix in {'.py', '.json', '.csv', '.md', '.txt', '.cff', '.yml'}:
            content = path.read_text(encoding='utf-8-sig')
            assert not re.search(r'(?:[A-Z]:[\\/](?:Users|Khanh_Loc)|/(?:nfs|local/scratch|home)/)', content), path.relative_to(ROOT)
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)', content), path.relative_to(ROOT)
            if path.suffix == '.json':
                assert 'MODFLOW2000 Layer Property Flow' not in content, path.relative_to(ROOT)
                assert not re.search(r'OPEN/CLOSE\s+[\"\x27]', content), path.relative_to(ROOT)
                json.loads(content)
    cff = yaml.safe_load((ROOT/'CITATION.cff').read_text())
    assert cff['type'] == 'software' and cff['preferred-citation']['type'] == 'unpublished'
    assert 'doi' not in cff and 'version' not in cff and 'doi' not in cff['preferred-citation']
    snapshot = json.loads((ROOT/'docs/source_snapshot.json').read_text())
    preparation_edits = {'scripts/build_manifest.py'}
    for row in snapshot:
        path = ROOT/row['path']
        assert path.is_file(), row['path']
        if row['path'] not in preparation_edits:
            assert digest(path) == row['copied_sha256'], row['path']
    with (ROOT/'ARTIFACT_MAP.csv').open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            assert digest(ROOT/row['path']) == row['sha256'], row['path']
    split = json.loads((ROOT/'metadata/param_split_fixed.json').read_text())
    groups = [set(split['splits'][key]['param_ids']) for key in ['train', 'val', 'test']]
    assert all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i+1, 3))
    assert sum(map(len, groups)) == 162
    config = json.loads((ROOT/'configs/ta_anp_no_k.json').read_text())
    assert config['use_k'] is False and config['use_flow'] is True
    assert config['epochs'] == 150 and config['batch_size'] * config['grad_accum_steps'] == 16
    print(f'PASS: {parsed} Python sources, {len(files)} files, snapshot hashes, fixed split, noK protocol, citation and content scan')

if __name__ == '__main__':
    main()
