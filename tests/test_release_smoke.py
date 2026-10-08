from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.compute_plume_metrics import log_to_phys
from scripts.make_figures import load_ts
from scripts.paired_significance_tests import compute_statistics
from scripts.train_ta_anp import CFG, disable_auxiliary_losses
from src.models.ta_anp import GroundwaterTAANP


class ReleaseSmokeTests(unittest.TestCase):
    def test_no_auxiliary_loss_matches_archived_ablation(self) -> None:
        cfg = CFG()
        disable_auxiliary_losses(cfg)
        self.assertEqual(cfg.lambda_plume, 0.0)
        self.assertEqual(cfg.lambda_centroid, 0.0)
        self.assertEqual(cfg.lambda_mass, 0.0)

    def test_horizon_loader_accepts_current_summed_concentration_columns(self) -> None:
        content = (
            "future_timestep,summed_conc_abs_error_mean,summed_conc_abs_error_std\n"
            "15,10.0,1.0\n16,12.0,2.0\n"
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            path.write_text(content, encoding="utf-8")
            timesteps, get = load_ts(path)
            np.testing.assert_array_equal(timesteps, np.asarray([15, 16]))
            np.testing.assert_allclose(get("summed_conc_abs_error_mean"), [10.0, 12.0])

    def test_significance_output_separates_requested_and_two_sided_p_values(self) -> None:
        left = {0: 0.8, 1: 0.9, 2: 1.0}
        right = {0: 0.7, 1: 0.75, 2: 0.8}
        result = compute_statistics(left, right, alternative="greater")
        self.assertEqual(result["alternative"], "greater")
        self.assertIn("wilcoxon_p_requested", result)
        self.assertIn("wilcoxon_p_two_sided", result)
        self.assertIn("paired_t_p_requested", result)
        self.assertIn("paired_t_p_two_sided", result)
        self.assertLessEqual(result["wilcoxon_p_requested"], result["wilcoxon_p_two_sided"])

    def test_ta_anp_forward_shapes(self) -> None:
        for use_k, k_fusion in [(False, "append"), (True, "append"), (True, "film")]:
            with self.subTest(use_k=use_k, k_fusion=k_fusion):
                model = GroundwaterTAANP(
                    hidden_channels=8,
                    token_dim=16,
                    latent_dim=4,
                    time_embed_dim=8,
                    num_heads=4,
                    use_k=use_k,
                    use_flow=True,
                    k_fusion=k_fusion,
                ).eval()
                c_obs = torch.full((2, 3, 16, 16), -8.0)
                k_field = torch.zeros((2, 1, 16, 16))
                times_obs = torch.tensor([[0.0, 0.2, 0.4], [0.0, 0.2, 0.4]])
                times_query = torch.tensor([[0.6, 0.8], [0.6, 0.8]])

                with torch.no_grad():
                    pred, aux = model(c_obs, k_field, times_obs, times_query, deterministic=True)

                self.assertEqual(pred.shape, (2, 2, 16, 16))
                self.assertTrue(torch.isfinite(pred).all())
                self.assertEqual(aux["prior_mu"].shape, (2, 4))

    def test_log_to_phys_clips_roundoff_below_zero(self) -> None:
        log_c = np.asarray([-12.0, -8.0, -2.0], dtype=np.float32)
        physical = log_to_phys(log_c, eps_c=1e-12)
        self.assertTrue(np.all(physical >= 0.0))
        self.assertAlmostEqual(float(physical[0]), 0.0, places=16)
        self.assertAlmostEqual(float(physical[1]), 1e-8 - 1e-12, places=12)


if __name__ == "__main__":
    unittest.main()
