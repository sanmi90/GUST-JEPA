"""Tests for the near-body lift-element observable targets (Session 34)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.lift_element import DEFAULT_ADJACENT_MASK_PATH
from src.data import nearbody_observables as nb

pytestmark = pytest.mark.skipif(
    not DEFAULT_ADJACENT_MASK_PATH.exists(),
    reason="airfoil_adjacent_mask.npy not present (outputs/ is gitignored)",
)


def _rand_field(*shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g)


class TestModeDim:
    def test_mode_output_dim(self):
        assert nb.mode_output_dim("nearbody_lift_element") == 80

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            nb.mode_output_dim("bogus")
        with pytest.raises(ValueError):
            nb.compute_nearbody_observable(_rand_field(2, 192, 96), "bogus")


class TestBand:
    def test_band_roi_nontrivial(self):
        r0, r1, c0, c1 = nb.nearbody_roi_window()
        assert 10 < (r1 - r0) < 100
        assert 10 < (c1 - c0) < 60

    def test_band_tensor_shape_and_range(self):
        band = nb.get_nearbody_band_tensor()
        assert band.shape == (192, 96)
        assert float(band.min()) >= 0.0 and float(band.max()) <= 1.0


class TestTargets:
    def test_shapes_3d_and_4d(self):
        out3 = nb.nearbody_lift_element_target(_rand_field(3, 192, 96))
        assert out3.shape == (3, 80)
        out4 = nb.nearbody_lift_element_target(_rand_field(2, 3, 192, 96))
        assert out4.shape == (2, 3, 80)

    def test_sign_block_symmetry(self):
        field = _rand_field(4, 192, 96)
        t_pos = nb.nearbody_patch_signed_target(field)
        t_neg = nb.nearbody_patch_signed_target(-field)
        # Flipping the field sign swaps the positive and negative 32-D blocks.
        torch.testing.assert_close(t_pos[:, :32], t_neg[:, 32:64])
        torch.testing.assert_close(t_pos[:, 32:64], t_neg[:, :32])

    def test_spectrum_sign_invariant(self):
        field = _rand_field(4, 192, 96)
        s1 = nb.nearbody_radial_spectrum_target(field)
        s2 = nb.nearbody_radial_spectrum_target(-field)
        torch.testing.assert_close(s1, s2)

    def test_zero_input_gives_zero_target(self):
        out = nb.nearbody_lift_element_target(torch.zeros(2, 192, 96))
        torch.testing.assert_close(out, torch.zeros(2, 80))

    def test_dispatch_matches_direct(self):
        field = _rand_field(2, 192, 96)
        torch.testing.assert_close(
            nb.compute_nearbody_observable(field, "nearbody_lift_element"),
            nb.nearbody_lift_element_target(field),
        )


class TestStatsRoundtrip:
    def test_standardization_roundtrip(self):
        targets = [
            nb.nearbody_lift_element_target(_rand_field(30, 192, 96)).numpy(),
            nb.nearbody_lift_element_target(torch.randn(20, 192, 96)).numpy(),
        ]
        stats = nb.compute_standardization_from_targets(targets, "nearbody_lift_element")
        assert stats.mean.shape == (80,)
        stacked = torch.from_numpy(np.concatenate(targets, axis=0))
        z = stats.standardize(stacked)
        assert abs(float(z.mean())) < 1e-3
        # Constant dims (e.g. permanently empty patches) standardize to 0, so
        # check per-dim std only where the raw std is meaningfully nonzero.
        live = stats.std > 1e-6
        assert live.sum() > 40
        z_live = z[:, torch.from_numpy(live)]
        assert float((z_live.std(dim=0) - 1.0).abs().max()) < 0.05

    def test_stats_dict_roundtrip(self):
        targets = [nb.nearbody_lift_element_target(_rand_field(10, 192, 96)).numpy()]
        stats = nb.compute_standardization_from_targets(targets, "nearbody_lift_element")
        restored = nb.WakeObservableStats.from_dict(stats.to_dict())
        np.testing.assert_allclose(restored.mean, stats.mean)
        np.testing.assert_allclose(restored.std, stats.std)
