"""Cell-centred finite volumes and backward Euler for heat conduction."""

from dataclasses import dataclass
from math import fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import splu

from .boundaries import Boundaries, Kind, Side
from .geometry import Grid
from .materials import Material
from .validation import (
    IncompatibleSteadyError,
    InputError,
    NonUniqueSteadyError,
    SolverError,
    integer,
    real_number,
)

MAX_STEPS = 10_000
MAX_STORED_VALUES = 5_000_000
FloatArray = NDArray[np.float64]
NUMERICAL_ERROR = ("Numerical range exceeded. Check the geometry, properties and load. "
                   "(محدودهٔ محاسباتی رد شد. هندسه، خواص و بار حرارتی را بررسی کنید.)")


@dataclass(frozen=True)
class FaceFlux:
    """Fourier flux and integrated cross-section rates along +x and +y."""

    grid: Grid
    qx_w_m2: FloatArray  # [y centre, x face]
    qy_w_m2: FloatArray  # [y face, x centre]

    @property
    def x_rate_w(self) -> FloatArray:
        return self.qx_w_m2.sum(axis=0) * self.grid.x_face_area

    @property
    def y_rate_w(self) -> FloatArray:
        return self.qy_w_m2.sum(axis=1) * self.grid.y_face_area


@dataclass(frozen=True)
class Result:
    grid: Grid
    material: Material
    boundaries: Boundaries
    temperature_k: FloatArray  # [snapshot, y, x]
    inward_w: FloatArray  # [snapshot, side] in Side order
    balance_w: FloatArray  # net inward rate + generation - accumulation rate
    generation_w_m3: FloatArray  # [y, x], time independent
    generated_w: float
    stored_energy_j: FloatArray  # sensible energy relative to 0 K
    times_s: FloatArray | None = None
    storage_w: FloatArray | None = None
    reference_mean_k: float | None = None
    stored_energy_change_j: FloatArray | None = None

    @property
    def energy_change_j(self) -> FloatArray:
        """Energy relative to the initial field, preserving small transient changes."""
        if self.stored_energy_change_j is not None:
            return self.stored_energy_change_j
        return self.stored_energy_j - self.stored_energy_j[0]

    def face_flux(self, snapshot: int) -> FaceFlux:
        """Reconstruct signed flux on all faces using the assembled face laws."""
        grid = self.grid
        temperature = self.temperature_k[snapshot]
        k = float(self.material.conductivity_w_mk)
        qx = np.empty((grid.ny, grid.nx + 1))
        qy = np.empty((grid.ny + 1, grid.nx))
        qx[:, 1:-1] = -k * (np.diff(temperature, axis=1) / grid.dx)
        qy[1:-1, :] = -k * (np.diff(temperature, axis=0) / grid.dy)
        faces = _faces(grid, self.material, self.boundaries, np.arange(grid.size).reshape(grid.shape))
        for target, face, sign in zip(
            (qx[:, 0], qx[:, -1], qy[0, :], qy[-1, :]), faces, (1, -1, 1, -1), strict=True
        ):
            target[:] = sign * face.rate(temperature.ravel(), 0.0) / face.area
        _finite(qx, qy)
        return FaceFlux(grid, qx, qy)


@dataclass(frozen=True)
class _Face:
    cells: NDArray[np.intp]
    area: float
    conductance: float
    ambient_k: float
    prescribed_w: float

    def rate(self, relative_k: FloatArray, offset_k: float) -> FloatArray:
        return self.prescribed_w + self.conductance * (
            (self.ambient_k - offset_k) - relative_k[self.cells]
        )


def _positive_scale(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise SolverError(NUMERICAL_ERROR)
    return value


def _finite(*values: ArrayLike) -> None:
    if any(not np.all(np.isfinite(value)) for value in values):
        raise SolverError(NUMERICAL_ERROR)


def _faces(grid: Grid, material: Material, boundaries: Boundaries,
           ids: NDArray[np.intp]) -> list[_Face]:
    k = float(material.conductivity_w_mk)
    locations = (
        (ids[:, 0], grid.x_face_area, grid.dx),
        (ids[:, -1], grid.x_face_area, grid.dx),
        (ids[0, :], grid.y_face_area, grid.dy),
        (ids[-1, :], grid.y_face_area, grid.dy),
    )
    faces = []
    for side, (cells, area, spacing) in zip(Side, locations, strict=True):
        bc = boundaries[side]
        if bc.kind is Kind.TEMPERATURE:
            conductance = _positive_scale(k * (area / spacing) * 2)
            face = _Face(cells, area, conductance, bc.value, 0.0)
        elif bc.kind is Kind.CONVECTION:
            resistance = (spacing / k) / 2 + 1 / float(bc.h_w_m2k)
            conductance = _positive_scale(area / _positive_scale(resistance))
            face = _Face(cells, area, conductance, bc.value, 0.0)
        else:
            face = _Face(cells, area, 0.0, 0.0, bc.value * area)
        if not np.isfinite(face.prescribed_w):
            raise SolverError(NUMERICAL_ERROR)
        faces.append(face)
    return faces


def _assemble(grid: Grid, material: Material, boundaries: Boundaries,
              offset_k: float, source_w: FloatArray) -> tuple[csc_matrix, FloatArray, list[_Face]]:
    ids = np.arange(grid.size).reshape(grid.shape)
    left, right = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    bottom, top = ids[:-1, :].ravel(), ids[1:, :].ravel()
    k = float(material.conductivity_w_mk)
    gx = _positive_scale(k * (grid.x_face_area / grid.dx)) if left.size else 0.0
    gy = _positive_scale(k * (grid.y_face_area / grid.dy)) if bottom.size else 0.0
    first = np.concatenate((left, bottom))
    second = np.concatenate((right, top))
    link = np.concatenate((np.full(left.size, gx), np.full(bottom.size, gy)))

    diagonal = np.zeros(grid.size)
    np.add.at(diagonal, first, link)
    np.add.at(diagonal, second, link)
    rhs = source_w.copy()
    faces = _faces(grid, material, boundaries, ids)
    for face in faces:
        diagonal[face.cells] += face.conductance
        rhs[face.cells] += face.prescribed_w + face.conductance * (face.ambient_k - offset_k)

    flat_ids = ids.ravel()
    rows = np.concatenate((first, second, flat_ids))
    cols = np.concatenate((second, first, flat_ids))
    data = np.concatenate((-link, -link, diagonal))
    _finite(data, rhs)
    matrix = coo_matrix((data, (rows, cols)), shape=(grid.size, grid.size)).tocsc()
    return matrix, rhs, faces


def _factor(matrix: csc_matrix):
    _finite(matrix.data)
    try:
        return splu(matrix)
    except RuntimeError as exc:
        raise SolverError("Sparse factorization failed. Check the input scales. "
                          "(تجزیهٔ ماتریس انجام نشد. مقیاس ورودی‌ها را بررسی کنید.)") from exc


def _side_rates(faces: list[_Face], relative_k: FloatArray, offset_k: float) -> FloatArray:
    rates = np.array([np.sum(face.rate(relative_k, offset_k)) for face in faces])
    _finite(rates)
    return rates


def solve_steady(grid: Grid, material: Material, boundaries: Boundaries, *,
                 reference_mean_k: float | None = None,
                 generation_w_m3: ArrayLike = 0.0) -> Result:
    """Compute the steady field with a scalar or cellwise volumetric source."""
    anchor = next((boundaries[side] for side in Side if boundaries[side].anchors_temperature), None)
    if anchor is not None and reference_mean_k is not None:
        raise InputError("Use a reference mean only with flux or insulation on every side. "
                         "(دمای مرجع فقط برای چهار ضلع شار یا عایق کاربرد دارد.)")

    generation, source_w, generated_w = _volumetric_source(grid, generation_w_m3)
    offset_k = float(anchor.value) if anchor is not None else 0.0
    matrix, rhs, faces = _assemble(grid, material, boundaries, offset_k, source_w)
    if anchor is None:
        specified = np.concatenate((_side_rates(faces, np.zeros(grid.size), offset_k), source_w))
        scale = np.max(np.abs(specified))
        if scale:
            normalized = specified / scale
            if abs(fsum(normalized)) > 1e-12 * np.sum(np.abs(normalized)):
                raise IncompatibleSteadyError(
                    "Steady flux-only conditions require zero net heat input including generation. "
                    "(مجموع توان ورودی مرزی و توان تولیدی باید صفر باشد.)"
                )
        if reference_mean_k is None:
            raise NonUniqueSteadyError(
                "Set the mean temperature to select a unique steady flux-only solution. "
                "(برای تعیین جواب یکتا، دمای میانگین را وارد کنید.)"
            )
        offset_k = real_number(reference_mean_k, "Mean temperature [K] (دمای میانگین)", minimum=0)
        # Pin one cell, then recover the specified volume mean by a constant shift.
        relative_k = np.zeros(grid.size)
        if grid.size > 1:
            relative_k[1:] = _factor(matrix[1:, 1:]).solve(rhs[1:])
        relative_k -= relative_k.mean()
    else:
        relative_k = _factor(matrix).solve(rhs)

    temperature = (relative_k + offset_k).reshape((1, *grid.shape))
    rates = _side_rates(faces, relative_k, offset_k)[None, :]
    balance = rates.sum(axis=1) + generated_w
    energy = _cell_capacity(grid, material) * temperature.sum(axis=(1, 2))
    _finite(temperature, balance, energy)
    return Result(grid, material, boundaries, temperature, rates, balance, generation,
                  generated_w, energy, reference_mean_k=offset_k if anchor is None else None)


def _grid_field(grid: Grid, values: ArrayLike, label: str) -> FloatArray:
    message = (f"{label} must be a finite real scalar or an array of shape (ny, nx). "
               "(مقدار باید یک عدد حقیقی متناهی یا آرایه‌ای هم‌اندازهٔ شبکه باشد.)")
    try:
        source = np.asarray(values)
        if source.dtype.kind not in "fiu":
            raise TypeError("Expected a real-valued input")
        field = source.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(message) from exc
    if field.ndim == 0:
        field = np.full(grid.shape, float(field))
    if field.shape != grid.shape or not np.all(np.isfinite(field)):
        raise InputError(message)
    return field


def _volumetric_source(grid: Grid, values: ArrayLike) -> tuple[FloatArray, FloatArray, float]:
    generation = _grid_field(grid, values, "Heat generation [W/m³] (تولید حرارت)")
    source_w = generation.ravel() * grid.cell_volume
    generated_w = float(np.sum(source_w))
    _finite(source_w, generated_w)
    return generation, source_w, generated_w


def _cell_capacity(grid: Grid, material: Material) -> float:
    return _positive_scale(
        float(material.density_kg_m3) * float(material.specific_heat_j_kgk) * grid.cell_volume
    )


def solve_transient(grid: Grid, material: Material, boundaries: Boundaries, *,
                    initial_k: ArrayLike, duration_s: float, steps: int,
                    generation_w_m3: ArrayLike = 0.0) -> Result:
    """Advance with fixed time steps, storing the initial field and every time level."""
    duration = real_number(duration_s, "Duration [s] (مدت)", minimum=0, strict=True)
    steps = integer(steps, "Time steps (گام زمانی)", 1, MAX_STEPS)
    if grid.size * (steps + 1) > MAX_STORED_VALUES:
        raise InputError("Reduce the mesh size or number of time steps. "
                         "(اندازهٔ شبکه یا تعداد گام‌های زمانی را کاهش دهید.)")
    initial = _grid_field(grid, initial_k, "Initial temperature [K] (دمای اولیه)").ravel()
    if np.any(initial < 0):
        raise InputError("Initial temperature must be ≥ 0 K. (دمای اولیه باید نامنفی باشد.)")
    generation, source_w, generated_w = _volumetric_source(grid, generation_w_m3)
    dt = _positive_scale(duration / steps)
    capacity = _cell_capacity(grid, material)
    storage_coefficient = _positive_scale(capacity / dt)
    offset_k = float(initial[0])
    matrix, rhs, faces = _assemble(grid, material, boundaries, offset_k, source_w)
    implicit_matrix = matrix + diags(np.full(grid.size, storage_coefficient), format="csc")
    lu = _factor(implicit_matrix)

    temperature = np.empty((steps + 1, *grid.shape))
    rates = np.empty((steps + 1, len(Side)))
    storage = np.full(steps + 1, np.nan)
    energy_change = np.zeros(steps + 1)
    correction_j = 0.0
    relative_k = initial - offset_k
    temperature[0] = initial.reshape(grid.shape)
    rates[0] = _side_rates(faces, relative_k, offset_k)

    for step in range(1, steps + 1):
        # The increment form avoids multiplying the absolute temperature by C/dt.
        forcing = rhs - matrix @ relative_k
        _finite(forcing)
        increment_k = lu.solve(forcing)
        updated_k = relative_k + increment_k
        absolute_k = updated_k + offset_k
        _finite(absolute_k)
        temperature[step] = absolute_k.reshape(grid.shape)
        rates[step] = _side_rates(faces, updated_k, offset_k)
        summed_increment = np.sum(increment_k)
        storage[step] = storage_coefficient * summed_increment
        # Accumulate the small changes before adding any absolute temperature offset.
        change_j = capacity * summed_increment - correction_j
        energy_change[step] = energy_change[step - 1] + change_j
        correction_j = (energy_change[step] - energy_change[step - 1]) - change_j
        relative_k = updated_k

    balance = rates.sum(axis=1) + generated_w - storage
    energy = capacity * temperature.sum(axis=(1, 2))
    _finite(storage[1:], balance[1:], energy, energy_change)
    return Result(grid, material, boundaries, temperature, rates, balance, generation, generated_w, energy,
                  times_s=np.linspace(0.0, duration, steps + 1), storage_w=storage,
                  stored_energy_change_j=energy_change)
