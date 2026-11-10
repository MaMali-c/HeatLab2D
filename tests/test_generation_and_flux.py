"""Analytical and conservation tests for generation, face flux and energy."""

import csv
import json

import numpy as np
import pytest

from heatlab.boundaries import Boundaries, Boundary, Kind
from heatlab.geometry import Grid
from heatlab.materials import Material
from heatlab.report import face_flux_csv, history_csv, report_json
from heatlab.solver import solve_steady, solve_transient
from heatlab.validation import IncompatibleSteadyError, InputError, NonUniqueSteadyError
from heatlab.verification import energy_study, spatial_study, temporal_study

MATERIAL = Material(12, 2400, 850)
INSULATED = Boundary(Kind.INSULATED)
ALL_INSULATED = Boundaries(INSULATED, INSULATED, INSULATED, INSULATED)


@pytest.mark.parametrize("thickness", [0.01, 0.03])
def test_generated_heat_slab_face_flux_and_thickness(thickness):
    grid = Grid(0.2, 0.1, thickness, 20, 10)
    cold = Boundary(Kind.TEMPERATURE, 300)
    bc = Boundaries(cold, cold, INSULATED, INSULATED)
    source = 24_000
    result = solve_steady(grid, MATERIAL, bc, generation_w_m3=source)
    flux = result.face_flux(0)
    assert flux.qx_w_m2.shape == (10, 21)
    assert flux.qy_w_m2.shape == (11, 20)
    expected = source * (grid.x_faces_m - grid.width_m / 2)
    np.testing.assert_allclose(flux.qx_w_m2, np.broadcast_to(expected, flux.qx_w_m2.shape),
                               atol=1e-8, rtol=0)
    np.testing.assert_allclose(flux.qy_w_m2, 0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(flux.x_rate_w, expected * grid.height_m * thickness, atol=1e-10)
    power = source * grid.width_m * grid.height_m * thickness
    assert result.generated_w == pytest.approx(power)
    np.testing.assert_allclose(result.inward_w[0], [-power / 2, -power / 2, 0, 0], atol=1e-10)
    assert abs(result.balance_w[0]) < 1e-10


def test_one_cell_four_boundary_faces_with_generation():
    grid = Grid(0.2, 0.1, 0.02, 1, 1)
    bc = Boundaries(Boundary(Kind.TEMPERATURE, 350), Boundary(Kind.CONVECTION, 290, 30),
                    Boundary(Kind.FLUX, 120), Boundary(Kind.FLUX, -250))
    result = solve_steady(grid, MATERIAL, bc, generation_w_m3=25_000)
    left_g = 2 * 12 * grid.x_face_area / grid.dx
    right_g = grid.x_face_area / (grid.dx / (2 * 12) + 1 / 30)
    source_w = 25_000 * grid.cell_volume
    expected = (left_g * 350 + right_g * 290 + (120 - 250) * grid.y_face_area
                + source_w) / (left_g + right_g)
    assert result.temperature_k.item() == pytest.approx(expected)
    assert abs(result.balance_w.item()) < 1e-11
    flux = result.face_flux(0)
    np.testing.assert_allclose(flux.y_rate_w, np.array([120, 250]) * grid.y_face_area)
    np.testing.assert_allclose(flux.x_rate_w, [left_g * (350 - expected), right_g * (expected - 290)])


def test_flux_only_checks_source_compatibility_before_reference():
    grid = Grid(0.2, 0.1, 0.02, 10, 5)
    generation = 6000
    with pytest.raises(IncompatibleSteadyError):
        solve_steady(grid, MATERIAL, ALL_INSULATED, generation_w_m3=generation)
    removal = Boundary(Kind.FLUX, -generation * grid.width_m)
    bc = Boundaries(INSULATED, removal, INSULATED, INSULATED)
    with pytest.raises(NonUniqueSteadyError):
        solve_steady(grid, MATERIAL, bc, generation_w_m3=generation)
    result = solve_steady(grid, MATERIAL, bc, generation_w_m3=generation, reference_mean_k=310)
    assert result.temperature_k.mean() == pytest.approx(310)
    assert result.inward_w[0, 1] == pytest.approx(-result.generated_w)
    np.testing.assert_allclose(result.face_flux(0).x_rate_w, generation * grid.x_faces_m
                               * grid.height_m * grid.thickness_m, atol=1e-10)


def test_nonuniform_generation_and_initial_state_match_two_cell_solution():
    grid = Grid(0.2, 0.1, 0.02, 2, 1)
    initial = np.array([[340.0, 300.0]])
    source = np.array([[48_000.0, -24_000.0]])
    result = solve_transient(grid, MATERIAL, ALL_INSULATED, initial_k=initial,
                             duration_s=30, steps=3, generation_w_m3=source)

    capacity = 2400 * 850 * grid.cell_volume
    conductance = 12 * grid.x_face_area / grid.dx
    expected_mean = initial.mean() + source.mean() * result.times_s / (2400 * 850)
    steady_difference = (source[0, 0] - source[0, 1]) * grid.cell_volume / (2 * conductance)
    decay = (1 + 10 * 2 * conductance / capacity) ** -np.arange(4)
    difference = steady_difference + (40 - steady_difference) * decay
    expected = expected_mean[:, None] + difference[:, None] * np.array([0.5, -0.5])
    np.testing.assert_allclose(result.temperature_k[:, 0, :], expected, atol=1e-11, rtol=0)
    np.testing.assert_allclose(result.stored_energy_j, 2 * capacity * expected_mean, atol=1e-10)
    np.testing.assert_allclose(result.balance_w[1:], 0, atol=1e-10)

    flux = result.face_flux(3)
    inward_per_cell = (-np.diff(flux.qx_w_m2, axis=1) * grid.x_face_area
                       - np.diff(flux.qy_w_m2, axis=0) * grid.y_face_area)
    expected_storage = capacity * (result.temperature_k[3] - result.temperature_k[2]) / 10
    np.testing.assert_allclose(inward_per_cell + source * grid.cell_volume, expected_storage,
                               atol=1e-10, rtol=0)


@pytest.mark.parametrize("source", [np.inf, True, 2 + 1j, [[1, 2, 3]], [[1, np.nan]]])
def test_rejects_invalid_generation(source):
    grid = Grid(0.2, 0.1, 0.02, 2, 1)
    with pytest.raises(InputError, match="Heat generation"):
        solve_transient(grid, MATERIAL, ALL_INSULATED, initial_k=300, duration_s=30, steps=3,
                        generation_w_m3=source)


def test_generated_heat_and_face_exports():
    grid = Grid(0.2, 0.1, 0.02, 2, 1)
    cold = Boundary(Kind.TEMPERATURE, 300)
    bc = Boundaries(cold, cold, INSULATED, INSULATED)
    result = solve_transient(grid, MATERIAL, bc, initial_k=300, duration_s=10, steps=2,
                             generation_w_m3=[[10_000, 20_000]])
    rows = list(csv.DictReader(face_flux_csv(result, 2).splitlines()))
    assert len(rows) == 7
    assert rows[0]["axis"] == "x"
    assert float(rows[0]["x_m"]) == 0
    assert float(rows[0]["heat_rate_w"]) == pytest.approx(result.inward_w[2, 0])
    assert rows[-1]["axis"] == "y"
    assert float(rows[-1]["y_m"]) == pytest.approx(grid.height_m)
    history = list(csv.DictReader(history_csv(result).splitlines()))
    assert float(history[2]["generated_w"]) == result.generated_w
    report = json.loads(report_json(result, 2))
    assert report["generation_w_m3"] == [[10_000, 20_000]]
    assert report["snapshot"]["stored_energy_j"] == result.stored_energy_j[2]


def test_independent_spatial_and_temporal_refinement():
    spatial = spatial_study()
    temporal = temporal_study()
    np.testing.assert_allclose([row["order"] for row in spatial[1:]], 2, atol=1e-6, rtol=0)
    assert all(0.95 < row["order"] < 1.05 for row in temporal[1:])


def test_integrated_energy_balance_under_mixed_conditions():
    energy = energy_study()
    assert energy["abs_integrated_balance_j"] < 1e-8
    assert energy["max_abs_step_balance_w"] < 1e-10
