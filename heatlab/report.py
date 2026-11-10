"""Export computed snapshots, time histories, and reproducible run metadata."""

import csv
import io
import json
from dataclasses import asdict
from itertools import repeat

import numpy as np

from .boundaries import Kind, Side
from .solver import Result
from .visualization import Cell, RunComparison, energy_history


def field_csv(result: Result, snapshot: int) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("x_m", "y_m", "temperature_k"))
    x = result.grid.x_m
    for y, row in zip(result.grid.y_m, result.temperature_k[snapshot], strict=True):
        writer.writerows(zip(x, repeat(y, len(x)), row, strict=True))
    return buffer.getvalue()


def face_flux_csv(result: Result, snapshot: int) -> str:
    grid = result.grid
    flux = result.face_flux(snapshot)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("axis", "x_m", "y_m", "flux_w_m2", "heat_rate_w"))
    for axis, x, y, values, area in (
        ("x", grid.x_faces_m, grid.y_m, flux.qx_w_m2, grid.x_face_area),
        ("y", grid.x_m, grid.y_faces_m, flux.qy_w_m2, grid.y_face_area),
    ):
        for y_value, row in zip(y, values, strict=True):
            writer.writerows((axis, x_value, y_value, q, q * area)
                             for x_value, q in zip(x, row, strict=True))
    return buffer.getvalue()


def history_csv(result: Result) -> str:
    if result.times_s is None or result.storage_w is None:
        raise ValueError("A time history requires a transient result.")
    field = result.temperature_k
    lows = field.min(axis=(1, 2))
    means = field.mean(axis=(1, 2))
    highs = field.max(axis=(1, 2))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow((
        "time_s", "min_k", "mean_k", "max_k",
        *(f"{side.name.lower()}_inward_w" for side in Side),
        "net_inward_w", "storage_w", "balance_w", "generated_w", "stored_energy_j",
        "stored_energy_change_j",
    ))
    changes = result.energy_change_j
    for i, time in enumerate(result.times_s):
        inward = result.inward_w[i]
        storage = None if i == 0 else result.storage_w[i]
        balance = None if i == 0 else result.balance_w[i]
        writer.writerow((time, lows[i], means[i], highs[i], *inward,
                         inward.sum(), storage, balance, result.generated_w,
                         result.stored_energy_j[i], changes[i]))
    return buffer.getvalue()


def probe_csv(result: Result, probes: tuple[Cell, ...]) -> str:
    if result.times_s is None:
        raise ValueError("Probe history requires a transient result.")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("time_s", "probe", "x_m", "y_m", "temperature_k"))
    x, y = result.grid.x_m, result.grid.y_m
    for frame, time in enumerate(result.times_s):
        for number, (j, i) in enumerate(probes, start=1):
            writer.writerow((time, f"P{number}", x[i], y[j], result.temperature_k[frame, j, i]))
    return buffer.getvalue()


def energy_csv(result: Result) -> str:
    energy = energy_history(result)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("time_s", "stored_change_j", "boundary_j", "generated_j",
                     "supplied_j", "residual_j"))
    writer.writerows(zip(result.times_s, energy.change_j, energy.boundary_j, energy.generated_j,
                         energy.supplied_j, energy.residual_j, strict=True))
    return buffer.getvalue()


def comparison_csv(comparison: RunComparison) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("time_s", "rms_delta_k", "max_abs_delta_k"))
    writer.writerows(zip(comparison.transient.times_s, comparison.rms_k, comparison.max_abs_k,
                         strict=True))
    return buffer.getvalue()


def _boundary_data(result: Result) -> dict:
    data = {}
    for side in Side:
        boundary = result.boundaries[side]
        fields = {"kind": boundary.kind.name.lower()}
        if boundary.kind is Kind.TEMPERATURE:
            fields["temperature_k"] = float(boundary.value)
        elif boundary.kind is Kind.CONVECTION:
            fields["ambient_temperature_k"] = float(boundary.value)
            fields["h_w_m2k"] = float(boundary.h_w_m2k)
        else:
            fields["inward_flux_w_m2"] = float(boundary.value)
        data[side.name.lower()] = fields
    return data


def _scalar_or_grid(values: np.ndarray) -> float | list:
    first = values.flat[0]
    return float(first) if np.all(values == first) else values.tolist()


def report_json(result: Result, snapshot: int) -> str:
    transient = result.times_s is not None
    field = result.temperature_k[snapshot]
    balance = result.balance_w[snapshot]
    data = {
        "schema_version": 2,
        "mode": "transient" if transient else "steady",
        "spatial_method": "cell_centred_finite_volume",
        "grid": {
            "width_m": float(result.grid.width_m), "height_m": float(result.grid.height_m),
            "thickness_m": float(result.grid.thickness_m),
            "nx": int(result.grid.nx), "ny": int(result.grid.ny),
        },
        "material": {key: float(value) for key, value in asdict(result.material).items()},
        "boundaries": _boundary_data(result),
        "generation_w_m3": _scalar_or_grid(result.generation_w_m3),
        "energy_reference_k": 0.0,
        "snapshot": {
            "index": snapshot,
            "min_k": float(field.min()), "mean_k": float(field.mean()), "max_k": float(field.max()),
            "inward_w": {side.name.lower(): float(value) for side, value in
                         zip(Side, result.inward_w[snapshot], strict=True)},
            "net_inward_w": float(result.inward_w[snapshot].sum()),
            "generated_w": result.generated_w,
            "stored_energy_j": float(result.stored_energy_j[snapshot]),
            "balance_w": None if np.isnan(balance) else float(balance),
        },
    }
    if transient:
        times = result.times_s
        assert times is not None and result.storage_w is not None
        initial = result.temperature_k[0]
        data["time"] = {
            "method": "backward_euler", "duration_s": float(times[-1]),
            "steps": len(times) - 1, "step_s": float(times[1]),
            "initial_temperature_k": _scalar_or_grid(initial),
            "max_abs_balance_w": float(np.max(np.abs(result.balance_w[1:]))),
        }
        data["snapshot"]["time_s"] = float(times[snapshot])
        data["snapshot"]["stored_energy_change_j"] = float(result.energy_change_j[snapshot])
        data["snapshot"]["storage_w"] = None if snapshot == 0 else float(result.storage_w[snapshot])
    elif result.reference_mean_k is not None:
        data["reference_mean_k"] = float(result.reference_mean_k)

    return json.dumps(data, indent=2, allow_nan=False)
