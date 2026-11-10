"""Analytical benchmarks, refinement sensitivity, and independent energy auditing."""

from dataclasses import dataclass
import json
from math import log2

import numpy as np

from .report import report_json
from .solver import Result
from .verification import (
    BASE_TIME_STEPS, MODE_AMPLITUDE_K, MODE_MEAN_K, SLAB_NX,
    generated_slab, mixed_boundary_benchmark, mode_benchmark,
)
from .visualization import energy_history

GRID_TOL_K = 0.01
TIME_TOL_K = 0.01
POWER_TOL_W = 1e-8
ENERGY_TOL_J = 1e-8
TIME_STEPS = (*BASE_TIME_STEPS, 160, 320, 640)


def error_metrics(numerical: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Equal-weight error norms on the same cell centres of a uniform mesh."""
    numerical, reference = np.asarray(numerical, dtype=float), np.asarray(reference, dtype=float)
    if numerical.shape != reference.shape or not numerical.size:
        raise ValueError("Compare nonempty fields evaluated at the same points.")
    error = np.abs(numerical - reference)
    if not np.all(np.isfinite(error)):
        raise ValueError("Error metrics require finite differences.")
    maximum = float(np.max(error))
    scaled = error / maximum if maximum else error
    return {
        "mae_k": maximum * float(np.mean(scaled)),
        "rmse_k": maximum * float(np.sqrt(np.mean(scaled**2))),
        "max_temperature_error_k": maximum,
    }


@dataclass(frozen=True)
class EnergyAudit:
    power_residual_w: np.ndarray
    storage_w: np.ndarray
    accumulated_residual_j: np.ndarray | None

    @property
    def max_abs_power_w(self) -> float:
        return float(np.nanmax(np.abs(self.power_residual_w)))

    @property
    def max_abs_energy_j(self) -> float | None:
        return (None if self.accumulated_residual_j is None else
                float(np.max(np.abs(self.accumulated_residual_j))))

    @property
    def passed(self) -> bool:
        energy = self.max_abs_energy_j
        return self.max_abs_power_w <= POWER_TOL_W and (energy is None or energy <= ENERGY_TOL_J)


def audit_energy(result: Result) -> EnergyAudit:
    """Recompute balances from exported side powers and stored energy changes."""
    input_w = result.inward_w.sum(axis=1) + result.generated_w
    if result.times_s is None:
        return EnergyAudit(input_w, np.zeros_like(input_w), None)
    storage = np.r_[np.nan, np.diff(result.energy_change_j) / np.diff(result.times_s)]
    return EnergyAudit(input_w - storage, storage, energy_history(result).residual_j)


@dataclass(frozen=True)
class ValidationRun:
    spatial: list[dict]
    temporal: list[dict]
    steady: Result
    steady_exact: np.ndarray
    transient: Result
    mode: np.ndarray
    continuum_decay: float
    fv_decay: float
    mixed: Result
    steady_energy: EnergyAudit
    transient_energy: EnergyAudit

    @property
    def transient_continuum(self) -> np.ndarray:
        return self.mode_field(self.transient.times_s[-1], self.continuum_decay)

    @property
    def transient_semidiscrete(self) -> np.ndarray:
        return self.mode_field(self.transient.times_s[-1], self.fv_decay)

    def mode_field(self, time_s: float, decay: float) -> np.ndarray:
        return MODE_MEAN_K + self.mode * np.exp(-decay * time_s)


def _adjacent_difference(current: np.ndarray, previous: np.ndarray | None, *,
                         refined_grid: bool = False) -> float | None:
    if previous is None:
        return None
    if refined_grid:
        # The coarse centre lies halfway between each pair of fine centres.
        ny, nx = previous.shape
        current = current.reshape(ny, 2, nx, 2).mean(axis=(1, 3))
    return float(np.max(np.abs(current - previous)))


def _row(metrics: dict, history: list[dict], adjacent: float | None, tolerance: float) -> dict:
    maximum = metrics["max_temperature_error_k"]
    return {
        **metrics,
        "order": log2(history[-1]["max_temperature_error_k"] / maximum) if history else None,
        "pair_max_delta_k": adjacent,
        "pair_tolerance_k": tolerance,
        "pair_pass": None if adjacent is None else adjacent <= tolerance,
    }


def run_validation() -> ValidationRun:
    """Run all benchmarks with the production solver and collect measured outputs."""
    spatial, temporal = [], []
    previous = None
    for nx in SLAB_NX:
        steady, steady_exact = generated_slab(nx)
        grid = steady.grid
        field = steady.temperature_k[0]
        metrics = error_metrics(field, steady_exact)
        adjacent = _adjacent_difference(field, previous, refined_grid=True)
        spatial.append({"nx": grid.nx, "ny": grid.ny, "dx_m": grid.dx, "dy_m": grid.dy,
                        **_row(metrics, spatial, adjacent, GRID_TOL_K),
                        "signed_power_residual_w": float(steady.balance_w[0])})
        previous = field

    previous = None
    for steps in TIME_STEPS:
        transient, mode, continuum_decay, fv_decay = mode_benchmark(steps)
        times = transient.times_s
        field = transient.temperature_k[-1]
        exact_fv = MODE_MEAN_K + mode * np.exp(-fv_decay * times[-1])
        exact_pde = MODE_MEAN_K + mode * np.exp(-continuum_decay * times[-1])
        metrics = error_metrics(field, exact_fv)
        continuum = error_metrics(field, exact_pde)
        adjacent = _adjacent_difference(field, previous)
        temporal.append({
            "steps": steps, "dt_s": float(times[1]),
            **_row(metrics, temporal, adjacent, TIME_TOL_K),
            **{f"continuum_{key}": value for key, value in continuum.items()},
        })
        previous = field

    mixed = mixed_boundary_benchmark()
    return ValidationRun(spatial, temporal, steady, steady_exact, transient, mode, continuum_decay,
                         fv_decay, mixed, audit_energy(steady), audit_energy(mixed))


def _case_metadata(result: Result) -> dict:
    data = json.loads(report_json(result, 0))
    data.pop("snapshot")
    data.pop("schema_version")
    data["linear_solver"] = "SciPy SuperLU sparse direct factorization"
    return data


def current_run_manifest(result: Result) -> dict:
    audit = audit_energy(result)
    data = json.loads(report_json(result, len(result.temperature_k) - 1))
    data["independent_energy_audit"] = {
        "max_abs_power_residual_w": audit.max_abs_power_w,
        "max_abs_integrated_residual_j": audit.max_abs_energy_j,
    }
    data["comparison_scope"] = "Energy audit and model record"
    return data


def validation_manifest(run: ValidationRun) -> dict:
    mixed_energy = energy_history(run.mixed)
    return {
        "schema_version": 1,
        "assumptions": [
            "Homogeneous isotropic material with constant properties",
            "Uniform temperature through thickness; two thickness faces are adiabatic",
            "Constant boundary data and volumetric sources; no radiation or contact resistance",
            "Error statistics are evaluated at corresponding cell centres",
        ],
        "references": {
            "authors": ["T. L. Bergman", "A. S. Lavine", "F. P. Incropera", "D. P. DeWitt"],
            "title": "Fundamentals of Heat and Mass Transfer",
            "edition": 8,
            "publisher": "https://catalogimages.wiley.com/images/db/pdf/9781119320432.toc.pdf",
            "sections": ["1.3.1", "2.1", "2.3", "2.4", "3.5.1", "4.2", "4.4.3", "4.5.2", "5.5"],
        },
        "models": {
            "A_steady_generated_slab": {
                **_case_metadata(run.steady), "refinement_nx": [row["nx"] for row in run.spatial],
                "analytic_reference": "300 + qdot*x*(Lx-x)/(2*k)",
            },
            "B_transient_insulated_mode": {
                **_case_metadata(run.transient), "refinement_steps": [row["steps"] for row in run.temporal],
                "initial_temperature_formula": (
                    f"{MODE_MEAN_K:g} + {MODE_AMPLITUDE_K:g}*cos(pi*x/Lx)*cos(pi*y/Ly)"
                ),
                "continuum_decay_s_inv": run.continuum_decay, "fv_decay_s_inv": run.fv_decay,
            },
            "C_transient_mixed_energy": _case_metadata(run.mixed),
        },
        "acceptance": {
            "grid_pair_max_delta_k": GRID_TOL_K, "time_pair_max_delta_k": TIME_TOL_K,
            "absolute_power_residual_w": POWER_TOL_W, "absolute_energy_residual_j": ENERGY_TOL_J,
            "scope": "Benchmark-specific demonstration thresholds",
        },
        "grid_convergence": run.spatial,
        "time_convergence": run.temporal,
        "energy_audit": {
            "steady": {
                "signed_boundary_inward_w": run.steady.inward_w[0].tolist(),
                "side_order": ["left", "right", "bottom", "top"],
                "generated_w": run.steady.generated_w,
                "signed_power_residual_w": float(run.steady_energy.power_residual_w[0]),
                "pass": run.steady_energy.passed,
            },
            "transient": {
                "boundary_energy_j": float(mixed_energy.boundary_j[-1]),
                "generated_energy_j": float(mixed_energy.generated_j[-1]),
                "stored_energy_change_j": float(mixed_energy.change_j[-1]),
                "final_signed_residual_j": float(mixed_energy.residual_j[-1]),
                "max_abs_power_residual_w": run.transient_energy.max_abs_power_w,
                "max_abs_energy_residual_j": run.transient_energy.max_abs_energy_j,
                "pass": run.transient_energy.passed,
            },
        },
    }
