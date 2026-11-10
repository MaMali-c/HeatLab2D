"""Analytical benchmarks and regression tests for the thermal model."""

import unittest

import numpy as np

from heatlab.boundaries import Boundaries, Boundary, Kind
from heatlab.geometry import Grid
from heatlab.materials import Material
from heatlab.solver import solve_steady, solve_transient
from heatlab.validation import InputError

INSULATED = Boundary(Kind.INSULATED)
ALL_INSULATED = Boundaries(INSULATED, INSULATED, INSULATED, INSULATED)


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(0.28, 0.12, 0.02, 14, 6)
        self.material = Material(12.0, 2400.0, 850.0)

    def test_dirichlet_linear_slab(self):
        bc = Boundaries(Boundary(Kind.TEMPERATURE, 370), Boundary(Kind.TEMPERATURE, 290),
                        INSULATED, INSULATED)
        result = solve_steady(self.grid, self.material, bc)
        expected = 370 - 80 * self.grid.x_m / self.grid.width_m
        np.testing.assert_allclose(result.temperature_k[0],
                                   np.broadcast_to(expected, self.grid.shape), atol=1e-10, rtol=0)
        power = 12 * 80 / self.grid.width_m * self.grid.height_m * self.grid.thickness_m
        np.testing.assert_allclose(result.inward_w[0], [power, -power, 0, 0], atol=1e-10)
        self.assertAlmostEqual(result.balance_w[0], 0, places=10)

    def test_convection_series_resistance(self):
        bc = Boundaries(Boundary(Kind.CONVECTION, 380, 28), Boundary(Kind.CONVECTION, 290, 15),
                        INSULATED, INSULATED)
        result = solve_steady(self.grid, self.material, bc)
        flux = 90 / (1 / 28 + self.grid.width_m / 12 + 1 / 15)
        expected = 380 - flux / 28 - flux * self.grid.x_m / 12
        np.testing.assert_allclose(result.temperature_k[0],
                                   np.broadcast_to(expected, self.grid.shape), atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.inward_w[0, :2],
                                   [flux * 0.12 * 0.02, -flux * 0.12 * 0.02], atol=1e-10)

    def test_steady_two_dimensional_fourier_solution(self):
        cold = Boundary(Kind.TEMPERATURE, 300)
        bc = Boundaries(cold, cold, cold, Boundary(Kind.TEMPERATURE, 400))
        errors = []
        for nx, ny in ((20, 12), (40, 24)):
            grid = Grid(0.2, 0.12, 0.015, nx, ny)
            x, y = np.meshgrid(grid.x_m, grid.y_m)
            exact = np.full(grid.shape, 300.0)
            for mode in range(1, 100, 2):
                a = mode * np.pi / grid.width_m
                # Stable form of sinh(a*y) / sinh(a*height).
                ratio = (np.exp(a * (y - grid.height_m)) * np.expm1(-2 * a * y)
                         / np.expm1(-2 * a * grid.height_m))
                exact += 400 / (mode * np.pi) * np.sin(a * x) * ratio
            # Measure away from the corner discontinuities in prescribed temperature.
            interior = (x >= 0.03) & (x <= 0.17) & (y >= 0.03) & (y <= 0.09)
            result = solve_steady(grid, self.material, bc)
            errors.append(np.max(np.abs(result.temperature_k[0][interior] - exact[interior])))
            self.assertAlmostEqual(result.balance_w[0], 0, places=9)
        self.assertLess(errors[1], 0.04)
        self.assertLess(errors[1], errors[0] / 3)

    def test_balanced_flux_in_both_axes(self):
        bc = Boundaries(*(Boundary(Kind.FLUX, q) for q in (180, -180, -70, 70)))
        result = solve_steady(self.grid, self.material, bc, reference_mean_k=315)
        x, y = np.meshgrid(self.grid.x_m, self.grid.y_m)
        expected = 315 - 180 / 12 * (x - 0.14) + 70 / 12 * (y - 0.06)
        np.testing.assert_allclose(result.temperature_k[0], expected, atol=1e-10, rtol=0)
        self.assertAlmostEqual(result.temperature_k.mean(), 315, places=11)
        self.assertAlmostEqual(result.balance_w[0], 0, places=12)

    def test_steady_flux_compatibility_uses_area(self):
        bc = Boundaries(Boundary(Kind.FLUX, 10), INSULATED, Boundary(Kind.FLUX, -10), INSULATED)
        with self.assertRaisesRegex(InputError, "zero net heat input"):
            solve_steady(self.grid, self.material, bc, reference_mean_k=300)
        with self.assertRaisesRegex(InputError, "mean temperature"):
            solve_steady(self.grid, self.material, ALL_INSULATED)

    def test_single_insulated_cell(self):
        grid = Grid(0.28, 0.12, 0.02, 1, 1)
        steady = solve_steady(grid, self.material, ALL_INSULATED, reference_mean_k=302)
        transient = solve_transient(grid, self.material, ALL_INSULATED, initial_k=302,
                                    duration_s=1.0, steps=3)
        np.testing.assert_array_equal(steady.temperature_k, 302)
        np.testing.assert_array_equal(transient.temperature_k, 302)

    def test_transient_prescribed_flux_conserves_energy(self):
        bc = Boundaries(INSULATED, INSULATED, INSULATED, Boundary(Kind.FLUX, 520))
        result = solve_transient(self.grid, self.material, bc, initial_k=300,
                                 duration_s=60, steps=12)
        net = 520 * 0.28 * 0.02
        total_capacity = 2400 * 850 * 0.28 * 0.12 * 0.02
        np.testing.assert_allclose(result.temperature_k.mean(axis=(1, 2)),
                                   300 + net * result.times_s / total_capacity,
                                   atol=1e-10, rtol=0)
        self.assertTrue(np.isnan(result.balance_w[0]))
        np.testing.assert_allclose(result.storage_w[1:], net, atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.balance_w[1:], 0, atol=1e-10, rtol=0)

    def test_transient_two_dimensional_cosine_mode(self):
        grid = self.grid
        mode = np.cos(np.pi * grid.y_m[:, None] / grid.height_m) * np.cos(
            np.pi * grid.x_m[None, :] / grid.width_m)
        result = solve_transient(grid, self.material, ALL_INSULATED, initial_k=300 + 30 * mode,
                                 duration_s=100, steps=20)
        alpha = 12 / (2400 * 850)
        # Eigenvalue of the discrete insulated Laplacian.
        decay = 4 * alpha * (
            (np.sin(np.pi / (2 * grid.nx)) / grid.dx) ** 2
            + (np.sin(np.pi / (2 * grid.ny)) / grid.dy) ** 2
        )
        factor = (1 + 5 * decay) ** -20
        np.testing.assert_allclose(result.temperature_k[-1], 300 + 30 * mode * factor,
                                   atol=1e-10, rtol=0)

    def test_refinement_toward_continuum_solution(self):
        errors = []
        alpha = 12 / (2400 * 850)
        duration = 180
        decay = alpha * np.pi**2 * (1 / 0.28**2 + 1 / 0.12**2)
        for refinement in (1, 2, 4):
            grid = Grid(0.28, 0.12, 0.02, 7 * refinement, 3 * refinement)
            x, y = np.meshgrid(grid.x_m, grid.y_m)
            initial_mode = 30 * np.cos(np.pi * x / 0.28) * np.cos(np.pi * y / 0.12)
            result = solve_transient(grid, self.material, ALL_INSULATED, initial_k=300 + initial_mode,
                                     duration_s=duration, steps=12 * refinement**2)
            exact = 300 + initial_mode * np.exp(-decay * duration)
            errors.append(np.max(np.abs(result.temperature_k[-1] - exact)))
        self.assertLess(errors[1], errors[0] / 2.5)
        self.assertLess(errors[2], errors[1] / 2.5)

    def test_transient_convection_single_cell(self):
        grid = Grid(0.28, 0.12, 0.02, 1, 1)
        bc = Boundaries(Boundary(Kind.CONVECTION, 400, 25), INSULATED, INSULATED, INSULATED)
        result = solve_transient(grid, self.material, bc, initial_k=300,
                                 duration_s=75, steps=3)
        g = 0.12 * 0.02 / (0.28 / (2 * 12) + 1 / 25)
        capacity = 2400 * 850 * 0.28 * 0.12 * 0.02
        expected = 400 - 100 * (1 + 25 * g / capacity) ** -np.arange(4)
        np.testing.assert_allclose(result.temperature_k[:, 0, 0], expected, atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.balance_w[1:], 0, atol=1e-10, rtol=0)

    def test_invalid_inputs(self):
        cases = (
            lambda: Grid(0, 0.12, 0.02, 14, 6),
            lambda: Grid(1, 1, 1, 300, 300),
            lambda: Material(12, float("nan"), 850),
            lambda: Material(10**400, 2400, 850),
            lambda: Boundary(Kind.CONVECTION, 300, 0),
            lambda: Boundary(Kind.TEMPERATURE, -1),
            lambda: solve_transient(self.grid, self.material, ALL_INSULATED,
                                    initial_k=np.zeros((2, 2)), duration_s=60, steps=12),
            lambda: solve_transient(self.grid, self.material, ALL_INSULATED,
                                    initial_k=300, duration_s=0, steps=12),
            lambda: solve_transient(self.grid, self.material, ALL_INSULATED,
                                    initial_k=300, duration_s=60, steps=True),
            lambda: solve_transient(Grid(1, 1, 1, 100, 100), self.material, ALL_INSULATED,
                                    initial_k=300, duration_s=60, steps=500),
        )
        for i, case in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(InputError):
                case()


if __name__ == "__main__":
    unittest.main()
