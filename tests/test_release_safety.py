"""Synthetic input regressions; no simulator fields or experiment results."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts.check_release import require
from scripts.paired_significance_tests import compute_statistics, load_values
from scripts.release_paths import PROTECTED, safe_output_directory

ROOT = Path(__file__).resolve().parents[1]


class OutputSafetyTests(unittest.TestCase):
    def test_root_and_ancestors_are_rejected(self):
        for output in [ROOT, ROOT.parent]:
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                safe_output_directory(ROOT, output)

    def test_all_source_directories_and_children_are_rejected(self):
        for name in PROTECTED:
            for output in [ROOT/name, ROOT/name/'nested']:
                with self.subTest(output=str(output)), self.assertRaises(ValueError):
                    safe_output_directory(ROOT, output)

    def test_ignored_outputs_and_external_temp_are_allowed(self):
        self.assertEqual(safe_output_directory(ROOT, ROOT/'outputs/reproduction'),
                         (ROOT/'outputs/reproduction').resolve())
        with TemporaryDirectory() as temp:
            self.assertEqual(safe_output_directory(ROOT, Path(temp)), Path(temp).resolve())

    def test_optimized_cli_cannot_write_to_metadata(self):
        run = subprocess.run([sys.executable, '-O', 'scripts/reproduce_summaries.py',
                              '--out-dir', 'metadata'], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('Refusing to overwrite', run.stderr)

    def test_optimized_release_check_still_raises(self):
        with self.assertRaises(ValueError):
            require(False, 'negative unit test')
        run = subprocess.run([sys.executable, '-O', '-c',
                              "from scripts.check_release import require; require(False, 'negative unit test')"],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('negative unit test', run.stderr)


class PairingSafetyTests(unittest.TestCase):
    def test_duplicate_seed_is_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'synthetic.csv'
            path.write_text('seed,value\n0,0.1\n0,0.2\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                load_values(str(path), 'seed', 'value')

    def test_nonfinite_table_values_are_rejected(self):
        for value in ['nan', 'inf', '-inf']:
            with self.subTest(value=value), TemporaryDirectory() as temp:
                path = Path(temp)/'synthetic.csv'
                path.write_text(f'seed,value\n0,{value}\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Non-finite'):
                    load_values(str(path), 'seed', 'value')

    def test_missing_paired_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same seed IDs'):
            compute_statistics({0: .1, 1: .2}, {0: .3, 1: .4, 2: .5}, 'two-sided')

    def test_nonfinite_function_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            compute_statistics({0: float('nan'), 1: .2}, {0: .3, 1: .4}, 'two-sided')


if __name__ == '__main__':
    unittest.main()
