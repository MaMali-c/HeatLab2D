"""Numerical data shared by the interactive and exported visualizations."""

from dataclasses import dataclass
from math import floor

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .solver import Result, solve_steady, solve_transient

THERMAL = (
    (0.0, "#122fc2"), (0.18, "#234df5"), (0.37, "#2278f5"),
    (0.50, "#1de2c5"), (0.64, "#8de766"), (0.76, "#ffe45a"),
    (0.88, "#ff8a35"), (1.0, "#ec393f"),
)
PROBE_COLORS = ("#FF6474", "#FFD166", "#50DDF7")
PLAYBACK_DURATION_S = 18.0
Cell = tuple[int, int]  # row, column
FloatArray = NDArray[np.float64]


def temperature_values(values, unit: str):
    return values - 273.15 if unit == "°C" else values


@dataclass(frozen=True)
class TemperatureScale:
    """Fixed range in kelvin, independent of the selected frame and display unit."""

    low_k: float
    high_k: float

    @classmethod
    def from_results(cls, *results: Result) -> "TemperatureScale":
        low = min(float(result.temperature_k.min()) for result in results)
        high = max(float(result.temperature_k.max()) for result in results)
        return cls(low, high) if low < high else cls(low - 0.5, high + 0.5)

    def limits(self, unit: str) -> tuple[float, float]:
        return temperature_values(self.low_k, unit), temperature_values(self.high_k, unit)

    def levels(self, unit: str) -> FloatArray:
        low, high = self.limits(unit)
        return np.linspace(low, high, 9)[1:-1]


def playback_frame(start: int, last: int, elapsed_s: float, speed: float) -> int:
    """Frame after elapsed wall time at a multiple of the nominal playback speed."""
    advance = floor(max(0.0, elapsed_s) * speed * last / PLAYBACK_DURATION_S)
    return min(last, start + advance)


def sampled_frames(count: int, limit: int) -> tuple[int, ...]:
    """Choose at most limit stored frames, including both endpoints."""
    if count < 2 or limit < 2:
        raise ValueError("Animation needs at least two frames.")
    return tuple(np.linspace(0, count - 1, min(count, limit), dtype=int).tolist())


@dataclass(frozen=True)
class EnergyHistory:
    change_j: FloatArray
    boundary_j: FloatArray
    generated_j: FloatArray

    @property
    def supplied_j(self) -> FloatArray:
        return self.boundary_j + self.generated_j

    @property
    def residual_j(self) -> FloatArray:
        return self.supplied_j - self.change_j


def energy_history(result: Result) -> EnergyHistory:
    """Integrate boundary rates with the same right-endpoint rule as Backward Euler."""
    times = result.times_s
    if times is None:
        raise ValueError("Energy history requires a transient result.")
    boundary_steps = np.diff(times) * result.inward_w[1:].sum(axis=1)
    return EnergyHistory(
        result.energy_change_j,
        np.r_[0.0, np.cumsum(boundary_steps)],
        result.generated_w * (times - times[0]),
    )


@dataclass(frozen=True)
class RunComparison:
    steady: Result
    transient: Result
    scale: TemperatureScale
    rms_k: FloatArray
    max_abs_k: FloatArray

    def difference_k(self, snapshot: int) -> FloatArray:
        return self.transient.temperature_k[snapshot] - self.steady.temperature_k[0]


def compare_results(steady: Result, transient: Result) -> RunComparison:
    """Compare solutions on matching cells under identical material and loads."""
    if steady.times_s is not None or transient.times_s is None:
        raise ValueError("A steady result and a transient result are required in that order.")
    if (steady.grid != transient.grid or steady.material != transient.material
            or steady.boundaries != transient.boundaries
            or not np.array_equal(steady.generation_w_m3, transient.generation_w_m3)):
        raise ValueError("The solutions must have identical geometry, material and loads.")
    reference = steady.temperature_k[0]
    rms, peak = np.empty(len(transient.times_s)), np.empty(len(transient.times_s))
    # Work one frame at a time to avoid a second copy of the full time history.
    for index, field in enumerate(transient.temperature_k):
        delta = field - reference
        peak[index] = np.max(np.abs(delta))
        rms[index] = peak[index] * np.sqrt(np.mean((delta / peak[index])**2)) if peak[index] else 0
    return RunComparison(steady, transient, TemperatureScale.from_results(steady, transient), rms, peak)


def solve_comparison(result: Result, *, initial_k: ArrayLike | None = None,
                     duration_s: float = 600.0, steps: int = 120) -> RunComparison:
    """Solve the other regime with the current result's model and cellwise source."""
    model = (result.grid, result.material, result.boundaries)
    source = result.generation_w_m3
    if result.times_s is None:
        if initial_k is None:
            initial_k = 298.15 if result.reference_mean_k is None else result.reference_mean_k
        transient = solve_transient(*model, initial_k=initial_k, duration_s=duration_s,
                                    steps=steps, generation_w_m3=source)
        return compare_results(result, transient)
    reference = None
    if not result.boundaries.has_temperature_anchor:
        reference = float(result.temperature_k[0].mean())
    steady = solve_steady(*model, reference_mean_k=reference, generation_w_m3=source)
    return compare_results(steady, result)
