"""Publication-style PNG figures generated from a ValidationRun."""

from io import BytesIO

import numpy as np

from .scientific_validation import GRID_TOL_K, TIME_TOL_K, ValidationRun
from .verification import MODE_MEAN_K
from .visualization import energy_history

NAVY = "#10253D"
TEAL = "#087F8C"
CORAL = "#B84D34"
MUTED = "#455B70"
GRID = "#DCE6EE"


def _setup():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.6, 3.18), dpi=190, facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(1, 2, gridspec_kw={"wspace": 0.12})


def _axis(ax, title: str, x: str, y: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=NAVY, fontweight="bold", pad=8)
    ax.set(xlabel=x, ylabel=y)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(MUTED)
        label.set_fontsize(8.5)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    for spine in ax.spines.values():
        spine.set_color("#BBCAD7")
    ax.grid(alpha=0.82, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _legend(ax, loc="best") -> None:
    ax.legend(loc=loc, fontsize=7.5, facecolor="white", framealpha=0.95,
              edgecolor=GRID, labelcolor=NAVY)


def _png(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=190, facecolor="white")
    return buffer.getvalue()


def steady_figure(run: ValidationRun) -> bytes:
    figure, (ax, error_ax) = _setup()
    result = run.steady
    grid = result.grid
    j = grid.ny // 2
    x = grid.x_m
    ax.plot(x, run.steady_exact[j], color=TEAL, lw=2, label="Analytical")
    ax.plot(x[::4], result.temperature_k[0, j, ::4], "o", markerfacecolor="white",
            color=NAVY, ms=3.2, label="Finite volume", zorder=3)
    _axis(ax, "A   Temperature at the mid-height row", "x [m]", "Temperature [K]")
    _legend(ax, "lower center")

    _axis(error_ax, "B   Signed error across the mesh", "x [m]", "y [m]")
    error_ax.grid(False)
    error = 1000 * (result.temperature_k[0] - run.steady_exact)
    mesh = error_ax.pcolormesh(grid.x_faces_m, grid.y_faces_m, error, cmap="Blues",
                               vmin=0, vmax=float(np.max(error)) * 1.08, shading="flat")
    error_ax.set_aspect("equal")
    bar = figure.colorbar(mesh, ax=error_ax, fraction=0.047, pad=0.04)
    bar.set_label("Error [mK]", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    return _png(figure)


def transient_figure(run: ValidationRun) -> bytes:
    figure, (ax, error_ax) = _setup()
    result = run.transient
    grid = result.grid
    times = result.times_s
    initial_amplitude = run.mode[0, 0]
    ax.plot(times, MODE_MEAN_K + initial_amplitude * np.exp(-run.continuum_decay * times),
            color=TEAL, lw=2, label="Continuum reference")
    ax.plot(times, MODE_MEAN_K + initial_amplitude * np.exp(-run.fv_decay * times),
            "--", color=CORAL, lw=1.2, label="Exact-in-time FV")
    ax.plot(times[::40], result.temperature_k[::40, 0, 0], "o", markerfacecolor="white",
            color=NAVY, ms=3, label="Backward Euler")
    _axis(ax, f"A   Cell at ({grid.x_m[0]:g}, {grid.y_m[0]:g}) m", "Time [s]", "Temperature [K]")
    _legend(ax, "upper right")

    _axis(error_ax, "B   Final error vs continuum", "x [m]", "y [m]")
    error_ax.grid(False)
    error = 1000 * (result.temperature_k[-1] - run.transient_continuum)
    bound = float(np.max(np.abs(error)))
    mesh = error_ax.pcolormesh(grid.x_faces_m, grid.y_faces_m, error, cmap="RdBu_r",
                               vmin=-bound, vmax=bound, shading="flat")
    error_ax.set_aspect("equal")
    bar = figure.colorbar(mesh, ax=error_ax, fraction=0.047, pad=0.04)
    bar.set_label("Error [mK]", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    return _png(figure)


def convergence_figure(run: ValidationRun) -> bytes:
    from matplotlib.ticker import NullFormatter

    figure, axes = _setup()
    for ax, study, key, title, xlabel, tolerance in (
        (axes[0], run.spatial, "dx_m", "A   Spatial refinement", "Cell spacing [m]", GRID_TOL_K),
        (axes[1], run.temporal, "dt_s", "B   Temporal refinement", "Time step [s]", TIME_TOL_K),
    ):
        x = [row[key] for row in study]
        ax.loglog(x, [row["max_temperature_error_k"] for row in study],
                  "o-", color=TEAL, ms=4, label="Max. error vs reference")
        ax.loglog(x[1:], [row["pair_max_delta_k"] for row in study[1:]],
                  "s--", color=NAVY, ms=3.2, label="Change vs preceding level")
        if key == "dt_s":
            ax.loglog(x, [row["continuum_max_temperature_error_k"] for row in study],
                      "^:", color=CORAL, ms=3.2, label="Max. error vs continuum")
        ax.axhline(tolerance, color="#81909D", lw=1, ls=":", label="Paired-change tolerance")
        ticks = x if key == "dx_m" else x[::2]
        ax.set_xticks(ticks, labels=[f"{value:.3g}" for value in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        _axis(ax, title, xlabel, "Temperature difference [K]")
        _legend(ax, "upper left")
    return _png(figure)


def energy_figure(run: ValidationRun) -> bytes:
    figure, (ax, error_ax) = _setup()
    result = run.mixed
    times = result.times_s
    energy = energy_history(result)
    ax.plot(times, energy.change_j, lw=2.4, color=NAVY, label="Stored change")
    ax.plot(times, energy.supplied_j, ":", lw=1.8, color=TEAL, label="Total supplied")
    ax.plot(times, energy.boundary_j, "--", lw=1.2, color=CORAL, label="Boundary input")
    ax.plot(times, energy.generated_j, lw=1.2, color="#8C682B", label="Generation")
    _axis(ax, "A   Integrated energy terms", "Time [s]", "Energy [J]")
    _legend(ax, "upper left")

    error_ax.axhline(0, color="#899DB0", lw=0.8)
    error_ax.plot(times[1:], 1e12 * run.transient_energy.power_residual_w[1:],
                  color=TEAL, lw=1.25, marker=".", ms=3)
    _axis(error_ax, "B   Independently recomputed residual", "Time [s]", "Power residual [pW]")
    return _png(figure)


def report_figures(run: ValidationRun) -> dict[str, bytes]:
    from .media import RENDER_LOCK

    with RENDER_LOCK:
        return {
            "steady_error.png": steady_figure(run),
            "transient_error.png": transient_figure(run),
            "convergence.png": convergence_figure(run),
            "energy_balance.png": energy_figure(run),
        }
