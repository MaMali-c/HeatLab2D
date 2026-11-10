"""Reproducible grid, time-step and energy checks. Run as a Python module."""

import json
from math import log2

import numpy as np

from .boundaries import Boundaries, Boundary, Kind
from .geometry import Grid
from .materials import Material
from .solver import Result, solve_steady, solve_transient

MATERIAL = Material(12.0, 2400.0, 850.0)
INSULATED = Boundary(Kind.INSULATED)
ALL_INSULATED = Boundaries(INSULATED, INSULATED, INSULATED, INSULATED)
SLAB_NX = (10, 20, 40, 80)
BASE_TIME_STEPS = (10, 20, 40, 80)
MODE_DURATION_S = 140.0
MODE_MEAN_K = 300.0
MODE_AMPLITUDE_K = 30.0


def generated_slab(nx: int) -> tuple[Result, np.ndarray]:
    """Solve the generated slab and evaluate its continuum solution at cell centres."""
    cold = Boundary(Kind.TEMPERATURE, 300.0)
    bc = Boundaries(cold, cold, INSULATED, INSULATED)
    generation = 24_000.0
    grid = Grid(0.2, 0.1, 0.02, nx, nx // 2)
    result = solve_steady(grid, MATERIAL, bc, generation_w_m3=generation)
    exact = 300.0 + generation * grid.x_m * (grid.width_m - grid.x_m) / (
        2 * MATERIAL.conductivity_w_mk
    )
    return result, np.broadcast_to(exact, grid.shape)


def mode_benchmark(steps: int) -> tuple[Result, np.ndarray, float, float]:
    """Solve the insulated cosine mode and return its continuum and FV decay rates."""
    grid = Grid(0.2, 0.1, 0.02, 20, 10)
    alpha = MATERIAL.conductivity_w_mk / (MATERIAL.density_kg_m3 * MATERIAL.specific_heat_j_kgk)
    discrete_decay = 4 * alpha * (
        (np.sin(np.pi / (2 * grid.nx)) / grid.dx) ** 2
        + (np.sin(np.pi / (2 * grid.ny)) / grid.dy) ** 2
    )
    continuous_decay = alpha * np.pi**2 * (grid.width_m**-2 + grid.height_m**-2)
    x_mode = np.cos(np.pi * grid.x_m / grid.width_m)
    y_mode = np.cos(np.pi * grid.y_m / grid.height_m)
    mode = MODE_AMPLITUDE_K * np.outer(y_mode, x_mode)
    result = solve_transient(grid, MATERIAL, ALL_INSULATED, initial_k=MODE_MEAN_K + mode,
                             duration_s=MODE_DURATION_S, steps=steps)
    return result, mode, float(continuous_decay), float(discrete_decay)


def mixed_boundary_benchmark() -> Result:
    grid = Grid(0.2, 0.1, 0.02, 20, 10)
    bc = Boundaries(Boundary(Kind.TEMPERATURE, 350), Boundary(Kind.CONVECTION, 290, 25),
                    INSULATED, Boundary(Kind.FLUX, 60))
    return solve_transient(grid, MATERIAL, bc, initial_k=300, duration_s=200.0,
                           steps=40, generation_w_m3=6000)


def spatial_study() -> list[dict]:
    """Compare the generated-heat slab solution on successively finer meshes."""
    rows = []
    for nx in SLAB_NX:
        result, exact = generated_slab(nx)
        grid = result.grid
        error = float(np.max(np.abs(result.temperature_k[0] - exact)))
        rows.append({
            "nx": grid.nx, "ny": grid.ny, "dx_m": grid.dx, "dy_m": grid.dy,
            "max_temperature_error_k": error,
            "order": log2(rows[-1]["max_temperature_error_k"] / error) if rows else None,
            "abs_balance_w": float(abs(result.balance_w[0])),
        })
    return rows


def temporal_study() -> list[dict]:
    """Use the exact-in-time discrete cosine mode to isolate temporal error."""
    rows = []
    for steps in BASE_TIME_STEPS:
        result, mode, _, decay = mode_benchmark(steps)
        exact = MODE_MEAN_K + mode * np.exp(-decay * MODE_DURATION_S)
        error = float(np.max(np.abs(result.temperature_k[-1] - exact)))
        rows.append({
            "steps": steps, "dt_s": MODE_DURATION_S / steps, "max_temperature_error_k": error,
            "order": log2(rows[-1]["max_temperature_error_k"] / error) if rows else None,
            "max_abs_balance_w": float(np.max(np.abs(result.balance_w[1:]))),
        })
    return rows


def energy_study() -> dict:
    """Check the stepwise power and integrated energy balance with mixed loads."""
    result = mixed_boundary_benchmark()
    duration = result.times_s[-1]
    boundary_energy = result.times_s[1] * result.inward_w[1:].sum()
    generated_energy = duration * result.generated_w
    energy_change = result.energy_change_j[-1]
    return {
        "boundary_energy_j": float(boundary_energy),
        "generated_energy_j": float(generated_energy),
        "stored_energy_change_j": float(energy_change),
        "abs_integrated_balance_j": float(abs(boundary_energy + generated_energy - energy_change)),
        "max_abs_step_balance_w": float(np.max(np.abs(result.balance_w[1:]))),
    }


def main() -> None:
    data = {
        "grid_convergence": spatial_study(),
        "time_convergence": temporal_study(),
        "transient_energy_balance": energy_study(),
    }
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
