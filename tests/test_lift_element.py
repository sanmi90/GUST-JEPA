"""Tests for the Chang lift-element machinery (Session 34, Track C)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.lift_element import (
    DX,
    DY,
    build_nearbody_band,
    lift_element_field,
    solve_phi_L,
)


def _square_solid(H: int = 64, W: int = 64, half: int = 4) -> np.ndarray:
    solid = np.zeros((H, W), dtype=bool)
    ci, cj = H // 2, W // 2
    solid[ci - half : ci + half, cj - half : cj + half] = True
    return solid


def _symmetric_square_solid(H: int, W: int, half: int) -> np.ndarray:
    """Square spanning [c-half, c+half] inclusive: mirror-symmetric about the
    center row/column of an odd-sized axis."""
    solid = np.zeros((H, W), dtype=bool)
    ci, cj = H // 2, W // 2
    solid[ci - half : ci + half + 1, cj - half : cj + half + 1] = True
    return solid


class TestSolvePhiL:
    def test_interior_residual_small(self):
        solid = _square_solid()
        res = solve_phi_L(solid, lift_dir=(0.0, 1.0), dx=0.05, dy=0.05)
        assert res["residual_linf"] < 1e-8

    def test_phi_zero_inside_solid_and_finite(self):
        solid = _square_solid()
        res = solve_phi_L(solid, lift_dir=(0.0, 1.0), dx=0.05, dy=0.05)
        assert np.all(res["phi"][solid] == 0.0)
        assert np.all(np.isfinite(res["phi"]))
        assert np.all(np.isfinite(res["grad_x"]))
        assert np.all(np.isfinite(res["grad_y"]))

    def test_nontrivial_solution(self):
        solid = _square_solid()
        res = solve_phi_L(solid, lift_dir=(0.0, 1.0), dx=0.05, dy=0.05)
        assert np.abs(res["phi"]).max() > 1e-6

    def test_dipole_antisymmetry_in_lift_direction(self):
        # For a y-symmetric solid centered in a y-symmetric domain and
        # e_L = (0, 1), phi_L is a y-dipole: phi(x, y) = -phi(x, -y).
        solid = _symmetric_square_solid(H=65, W=65, half=4)
        res = solve_phi_L(solid, lift_dir=(0.0, 1.0), dx=0.05, dy=0.05)
        phi = res["phi"]
        np.testing.assert_allclose(phi, -phi[:, ::-1], atol=1e-10)

    def test_lift_dir_sign_flip_flips_phi(self):
        solid = _square_solid()
        res_p = solve_phi_L(solid, lift_dir=(0.0, 1.0), dx=0.05, dy=0.05)
        res_m = solve_phi_L(solid, lift_dir=(0.0, -1.0), dx=0.05, dy=0.05)
        np.testing.assert_allclose(res_p["phi"], -res_m["phi"], atol=1e-12)

    def test_gradient_zero_inside_solid(self):
        solid = _square_solid()
        res = solve_phi_L(solid, lift_dir=(0.3, 0.9), dx=0.05, dy=0.05)
        assert np.all(res["grad_x"][solid] == 0.0)
        assert np.all(res["grad_y"][solid] == 0.0)


class TestLiftElementField:
    def test_flipping_all_fields_preserves_e(self):
        rng = np.random.default_rng(0)
        omega = rng.normal(size=(3, 8, 8)).astype(np.float32)
        u = rng.normal(size=(3, 8, 8)).astype(np.float32)
        v = rng.normal(size=(3, 8, 8)).astype(np.float32)
        gx = rng.normal(size=(8, 8)).astype(np.float32)
        gy = rng.normal(size=(8, 8)).astype(np.float32)
        e1 = lift_element_field(omega, u, v, gx, gy)
        e2 = lift_element_field(-omega, -u, -v, gx, gy)
        np.testing.assert_allclose(e1, e2, rtol=1e-6)

    def test_flipping_omega_only_flips_e(self):
        rng = np.random.default_rng(1)
        omega = rng.normal(size=(8, 8)).astype(np.float32)
        u = rng.normal(size=(8, 8)).astype(np.float32)
        v = rng.normal(size=(8, 8)).astype(np.float32)
        gx = rng.normal(size=(8, 8)).astype(np.float32)
        gy = rng.normal(size=(8, 8)).astype(np.float32)
        e1 = lift_element_field(omega, u, v, gx, gy)
        e2 = lift_element_field(-omega, u, v, gx, gy)
        np.testing.assert_allclose(e1, -e2, rtol=1e-6)

    def test_nan_fill(self):
        omega = np.full((4, 4), np.nan, dtype=np.float32)
        u = np.ones((4, 4), dtype=np.float32)
        v = np.ones((4, 4), dtype=np.float32)
        g = np.ones((4, 4), dtype=np.float32)
        e = lift_element_field(omega, u, v, g, g)
        assert np.all(e == 0.0)
        assert np.all(np.isfinite(e))


class TestNearbodyBand:
    def test_zero_on_masked_cells_and_far_field(self):
        mask = _square_solid(H=96, W=96, half=3)
        band = build_nearbody_band(mask, delta_n=0.3, dx=DX, dy=DY)
        assert np.all(band[mask] == 0.0)
        # Distance from the square to the domain corner far exceeds 0.3c.
        assert band[0, 0] == 0.0
        assert band[-1, -1] == 0.0

    def test_band_within_unit_interval_and_nonempty(self):
        mask = _square_solid(H=96, W=96, half=3)
        band = build_nearbody_band(mask, delta_n=0.3, dx=DX, dy=DY)
        assert band.min() >= 0.0 and band.max() <= 1.0
        assert (band > 0).sum() > 50

    def test_band_decays_with_distance(self):
        mask = _square_solid(H=96, W=96, half=3)
        band = build_nearbody_band(mask, delta_n=0.3, dx=DX, dy=DY)
        ci = 48
        # Walking away from the square along +x: adjacent pixel > farther pixel.
        j_edge = 48 + 3  # first fluid pixel east of the square
        assert band[ci, j_edge] > band[ci, j_edge + 4] > 0.0 or band[ci, j_edge + 4] == 0.0
        assert band[ci, j_edge] > 0.5

    def test_sign_symmetry_of_band(self):
        # The band depends only on geometry, so it is identical for both gust
        # signs by construction; check symmetry about the square's centerline.
        mask = _symmetric_square_solid(H=97, W=97, half=3)
        band = build_nearbody_band(mask, delta_n=0.3, dx=DX, dy=DY)
        np.testing.assert_allclose(band, band[:, ::-1], atol=1e-6)
