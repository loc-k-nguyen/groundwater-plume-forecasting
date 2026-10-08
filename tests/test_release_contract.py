from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.reproduce_summaries import build
from src.models.ta_anp import GroundwaterTAANP

ROOT = Path(__file__).resolve().parents[1]

class ReleaseContractTests(unittest.TestCase):
    def test_no_k_prediction_is_invariant_to_conductivity_input(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = GroundwaterTAANP(hidden_channels=8, token_dim=16, latent_dim=4,
                                time_embed_dim=8, num_heads=4, use_k=False).eval()
        history = torch.full((1, 3, 16, 16), -8.0)
        obs = torch.tensor([[0.0, 0.2, 0.4]])
        query = torch.tensor([[0.6, 0.8]])
        with torch.no_grad():
            a, _ = model(history, torch.zeros((1, 1, 16, 16)), obs, query, deterministic=True)
            b, _ = model(history, torch.randn((1, 1, 16, 16)), obs, query, deterministic=True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_grouped_summary_retains_twins_and_uncalibrated_intervals(self):
        result = build()
        self.assertEqual(result['conductivity_twins']['test_twinned'], 30)
        self.assertLess(result['prior_predictive']['coverage']['mean'], 0.5)
        self.assertAlmostEqual(result['grouped_split']['plume_ssim']['wilcoxon_two_sided_p'], 0.001953125)

    def test_manifest_detects_changed_payload(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'a.txt').write_text('original')
            script = str(ROOT/'scripts/build_manifest.py')
            subprocess.run([sys.executable, script, '--root', str(root)], check=True, capture_output=True)
            subprocess.run([sys.executable, script, '--root', str(root), '--check'], check=True, capture_output=True)
            (root/'a.txt').write_text('altered')
            check = subprocess.run([sys.executable, script, '--root', str(root), '--check'], capture_output=True)
            self.assertNotEqual(check.returncode, 0)

if __name__ == '__main__':
    unittest.main()
