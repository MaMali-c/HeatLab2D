"""Plotly views of calculated fields, face flux and measurement points."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .geometry import Grid
from .solver import FaceFlux, Result
from .visualization import (
    PROBE_COLORS,
    THERMAL,
    Cell,
    RunComparison,
    TemperatureScale,
    energy_history,
    temperature_values,
)

PLOT = "#0B1425"
TEXT = "#DFE9FA"
MUTED = "#A5B7D1"
GRID = "#273955"
FONT = "Vazirmatn, Arial, sans-serif"
PLOT_CONFIG = {
    "displaylogo": False, "responsive": True, "displayModeBar": "hover",
    "toImageButtonOptions": {"format": "png", "filename": "heatlab-figure", "scale": 2},
}


def default_probes(grid: Grid) -> tuple[Cell, ...]:
    row = nearest_index(grid.height_m / 2, grid.dy, grid.ny)
    return tuple((row, nearest_index(grid.width_m * fraction, grid.dx, grid.nx))
                 for fraction in (0.12, 0.50, 0.88))


def nearest_index(coordinate: float, spacing: float, count: int) -> int:
    return min(max(int(coordinate / spacing), 0), count - 1)


def cell_flux(flux: FaceFlux) -> tuple[np.ndarray, np.ndarray]:
    return ((flux.qx_w_m2[:, :-1] + flux.qx_w_m2[:, 1:]) / 2,
            (flux.qy_w_m2[:-1, :] + flux.qy_w_m2[1:, :]) / 2)


def _style(figure: go.Figure, height: int, *, right: int = 16) -> None:
    figure.update_layout(
        height=height, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT,
        font={"family": FONT, "size": 12, "color": TEXT},
        margin={"l": 48, "r": right, "t": 12, "b": 43},
        hoverlabel={"bgcolor": "#172338", "font_color": TEXT},
    )
    figure.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont_color=MUTED)
    figure.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont_color=MUTED)


def flux_vectors(grid: Grid, flux: FaceFlux) -> tuple[np.ndarray, ...]:
    """Sample display arrows without modifying the solved face fluxes."""
    qx, qy = cell_flux(flux)
    rows = np.unique(np.linspace(0, grid.ny - 1, min(grid.ny, 11), dtype=int))
    cols = np.unique(np.linspace(0, grid.nx - 1, min(grid.nx, 17), dtype=int))
    xs = grid.x_m[cols]
    ys = grid.y_m[rows]
    dx = qx[np.ix_(rows, cols)]
    dy = qy[np.ix_(rows, cols)]
    magnitude = np.hypot(dx, dy)
    peak = float(magnitude.max())
    if peak == 0:
        return tuple(np.empty(0) for _ in range(4))
    length = 0.70 * min(grid.width_m / len(cols), grid.height_m / len(rows))
    j, i = np.where(magnitude > 0.025 * peak)
    magnitude = magnitude[j, i]
    factor = length * np.sqrt(magnitude / peak) / magnitude
    return xs[i], ys[j], dx[j, i] * factor, dy[j, i] * factor


def _arrows(grid: Grid, flux: FaceFlux) -> go.Scatter | None:
    xs, ys, us, vs = flux_vectors(grid, flux)
    if not xs.size:
        return None
    ax, ay = [], []
    for x, y, u, v in zip(xs, ys, us, vs, strict=True):
        x0, y0 = x - 0.42 * u, y - 0.42 * v
        x1, y1 = x0 + u, y0 + v
        ax.extend((x0, x1, None, x1 - 0.28*u + 0.18*v, x1, x1 - 0.28*u - 0.18*v, None))
        ay.extend((y0, y1, None, y1 - 0.28*v - 0.18*u, y1, y1 - 0.28*v + 0.18*u, None))
    return go.Scatter(x=ax, y=ay, mode="lines", line={"color": "rgba(255,255,255,.75)", "width": 1.2},
                      name="Heat flux", hoverinfo="skip", showlegend=False)


def temperature_map(result: Result, snapshot: int, *, unit: str = "K",
                    probes: tuple[Cell, ...] = (), heatmap: bool = True, contours: bool = False,
                    flux: FaceFlux | None = None, picking: bool = False,
                    revision: int = 0, scale: TemperatureScale | None = None,
                    height: int = 428) -> go.Figure:
    grid = result.grid
    values = temperature_values(result.temperature_k[snapshot], unit)
    scale = scale or TemperatureScale.from_results(result)
    low, high = scale.limits(unit)
    figure = go.Figure()
    if heatmap:
        figure.add_trace(go.Heatmap(
            x0=grid.dx / 2, dx=grid.dx, y0=grid.dy / 2, dy=grid.dy, z=values,
            colorscale=THERMAL, zmin=low, zmax=high, zsmooth=False,
            colorbar={"title": {"text": f"T [{unit}]", "side": "top"}, "len": 0.87,
                      "thickness": 13, "outlinewidth": 0, "tickfont": {"color": MUTED}},
            hovertemplate=f"x %{{x:.4g}} m · y %{{y:.4g}} m<br>%{{z:.2f}} {unit}<extra></extra>",
        ))
    if contours and min(grid.shape) > 1 and np.ptp(values) > 1e-9:
        levels = scale.levels(unit)
        figure.add_trace(go.Contour(
            x=grid.x_m, y=grid.y_m, z=values, autocontour=False, showscale=False,
            contours={"coloring": "none", "showlabels": True, "start": levels[0],
                      "end": levels[-1], "size": levels[1] - levels[0],
                      "labelfont": {"color": "#fff", "size": 11}},
            line={"width": 1.15, "color": "rgba(255,255,255,.74)"}, hoverinfo="skip",
            showlegend=False,
        ))
    if flux is not None:
        arrows = _arrows(grid, flux)
        if arrows is not None:
            figure.add_trace(arrows)
    if probes:
        figure.add_trace(go.Scatter(
            x=[grid.x_m[i] for j, i in probes], y=[grid.y_m[j] for j, i in probes],
            text=[f"P{index + 1}" for index in range(len(probes))], mode="markers+text",
            textposition="top center", textfont={"color": "#fff", "size": 12},
            marker={"size": 10, "color": PROBE_COLORS[:len(probes)],
                    "line": {"color": "#fff", "width": 1.5}},
            customdata=[values[j, i] for j, i in probes], showlegend=False,
            hovertemplate=f"%{{text}} · %{{customdata:.2f}} {unit}<extra></extra>",
        ))
    if picking:
        rows, cols = np.indices(grid.shape)
        figure.add_trace(go.Scatter(
            x=grid.x_m[cols.ravel()], y=grid.y_m[rows.ravel()], mode="markers",
            marker={"size": 9, "color": "rgba(255,255,255,0.08)"},
            hoverinfo="none", showlegend=False, name="Measurement cells",
        ))
        figure.update_layout(clickmode="event+select")
    _style(figure, height, right=8)
    figure.update_layout(uirevision=f"map-{revision}", showlegend=False)
    figure.update_xaxes(title="x [m]", range=[0, grid.width_m], constrain="domain", nticks=6)
    figure.update_yaxes(title="y [m]", range=[0, grid.height_m], scaleanchor="x",
                        scaleratio=1, constrain="domain", nticks=6)
    return figure


def temperature_surface(result: Result, snapshot: int, unit: str, revision: int, *,
                        scale: TemperatureScale | None = None) -> go.Figure:
    grid = result.grid
    rows = np.unique(np.linspace(0, grid.ny - 1, min(grid.ny, 45), dtype=int))
    cols = np.unique(np.linspace(0, grid.nx - 1, min(grid.nx, 45), dtype=int))
    low, high = (scale or TemperatureScale.from_results(result)).limits(unit)
    values = temperature_values(result.temperature_k[snapshot][np.ix_(rows, cols)], unit)
    x, y = grid.x_m[cols], grid.y_m[rows]
    if x.size == 1 or y.size == 1:
        # A one-cell-wide solution represents a constant value across that cell.
        if x.size == 1:
            x, values = grid.x_faces_m, np.repeat(values, 2, axis=1)
        if y.size == 1:
            y, values = grid.y_faces_m, np.repeat(values, 2, axis=0)
    figure = go.Figure(go.Surface(
        x=x, y=y, z=values, colorscale=THERMAL, cmin=low, cmax=high, showscale=False,
        lighting={"ambient": 0.86, "diffuse": 0.6, "specular": 0.11},
        contours={"z": {"show": True, "usecolormap": False, "color": "#bdeaff", "width": 1}},
        hovertemplate=f"x %{{x:.3g}} m · y %{{y:.3g}} m<br>%{{z:.2f}} {unit}<extra></extra>",
    ))
    _style(figure, 261)
    axis = {"backgroundcolor": PLOT, "gridcolor": GRID, "zerolinecolor": GRID,
            "showbackground": True, "tickfont": {"size": 9, "color": MUTED}, "nticks": 4}
    figure.update_layout(
        margin={"l": 1, "r": 1, "t": 1, "b": 3}, uirevision=f"surface-{revision}",
        scene={"xaxis": {**axis, "title": "x [m]"}, "yaxis": {**axis, "title": "y [m]"},
               "zaxis": {**axis, "title": f"T [{unit}]", "range": [low, high]},
               "aspectmode": "manual", "aspectratio": {"x": 1.16, "y": 0.9, "z": 0.74},
               "camera": {"eye": {"x": 1.60, "y": -1.82, "z": 1.10}}},
    )
    return figure


def temperature_history(result: Result, probes: tuple[Cell, ...],
                        snapshot: int, unit: str) -> go.Figure:
    figure = go.Figure()
    for index, (j, i) in enumerate(probes):
        values = temperature_values(result.temperature_k[:, j, i], unit)
        figure.add_trace(go.Scatter(
            x=result.times_s, y=values, name=f"P{index + 1}", mode="lines",
            line={"color": PROBE_COLORS[index], "width": 2.4},
            hovertemplate=f"%{{x:.2f}} s · %{{y:.2f}} {unit}<extra>%{{fullData.name}}</extra>",
        ))
        figure.add_trace(go.Scatter(
            x=[result.times_s[snapshot]], y=[values[snapshot]], mode="markers", showlegend=False,
            marker={"color": PROBE_COLORS[index], "size": 7, "line": {"color": PLOT, "width": 1}},
            hoverinfo="skip",
        ))
    figure.add_vline(x=float(result.times_s[snapshot]), line_color="#8ea5c7", line_dash="dot")
    _style(figure, 230)
    figure.update_layout(
        xaxis_title="t [s]", yaxis_title=f"T [{unit}]",
        legend={"orientation": "h", "y": 1.04, "x": 1, "xanchor": "right",
                "yanchor": "bottom", "font": {"size": 11}},
        margin={"l": 50, "r": 12, "t": 28, "b": 42},
    )
    return figure


def heat_rate_history(result: Result, snapshot: int) -> go.Figure:
    if result.times_s is None:
        raise ValueError("Heat-rate history requires a transient solution.")
    figure = go.Figure()
    for label, values, color in (
        ("Boundary input (ورودی مرزی)", result.inward_w.sum(axis=1), "#22D3EE"),
        ("Generation (تولید حجمی)", np.full(len(result.times_s), result.generated_w), "#FFD166"),
        ("Accumulation (انباشت)", result.storage_w, "#AC8BFF"),
    ):
        figure.add_trace(go.Scatter(x=result.times_s, y=values, name=label, mode="lines",
                                    line={"color": color, "width": 2.2}))
    figure.add_vline(x=float(result.times_s[snapshot]), line_color="#8ea5c7", line_dash="dot")
    _style(figure, 275)
    figure.update_layout(xaxis_title="t [s]", yaxis_title="Power [W] (توان)",
                         legend={"orientation": "h", "y": 1.03, "yanchor": "bottom"},
                         margin={"l": 53, "r": 12, "t": 33, "b": 43})
    return figure


def energy_change_plot(result: Result, snapshot: int) -> go.Figure:
    energy = energy_history(result)
    figure = go.Figure()
    for label, values, color, dash in (
        ("Stored ΔE (تغییر انرژی ذخیره‌شده)", energy.change_j, "#22D3EE", "solid"),
        ("Total supplied (کل انرژی ورودی)", energy.supplied_j, "#F3F7FF", "dot"),
        ("Boundary input (ورودی مرزی)", energy.boundary_j, "#AC8BFF", "dash"),
        ("Generation (تولید حجمی)", energy.generated_j, "#FFD166", "dash"),
    ):
        figure.add_trace(go.Scatter(x=result.times_s, y=values, name=label, mode="lines",
                                    line={"color": color, "width": 2.3, "dash": dash}))
    figure.add_vline(x=float(result.times_s[snapshot]), line_color="#8ea5c7", line_dash="dot")
    _style(figure, 284)
    figure.update_layout(xaxis_title="t [s]", yaxis_title="ΔE [J]",
                         legend={"orientation": "h", "y": 1.04, "yanchor": "bottom"},
                         margin={"l": 58, "r": 12, "t": 44, "b": 43})
    return figure


def temperature_difference(comparison: RunComparison, snapshot: int) -> go.Figure:
    grid = comparison.steady.grid
    peak = float(comparison.max_abs_k.max()) or 0.5
    figure = go.Figure(go.Heatmap(
        x0=grid.dx / 2, dx=grid.dx, y0=grid.dy / 2, dy=grid.dy,
        z=comparison.difference_k(snapshot), colorscale="RdBu_r",
        zmin=-peak, zmax=peak, zsmooth=False,
        colorbar={"title": "ΔT [K]", "len": 0.86, "thickness": 13, "outlinewidth": 0},
        hovertemplate="x %{x:.4g} m · y %{y:.4g} m<br>ΔT %{z:+.3f} K<extra></extra>",
    ))
    _style(figure, 315, right=8)
    figure.update_xaxes(title="x [m]", range=[0, grid.width_m], constrain="domain", nticks=6)
    figure.update_yaxes(title="y [m]", range=[0, grid.height_m], scaleanchor="x",
                        scaleratio=1, constrain="domain", nticks=6)
    return figure


def comparison_history(comparison: RunComparison, snapshot: int) -> go.Figure:
    figure = go.Figure()
    for name, values, color in (
        ("RMS ΔT (اختلاف مؤثر)", comparison.rms_k, "#22D3EE"),
        ("Max |ΔT| (بیشینهٔ اختلاف)", comparison.max_abs_k, "#FFD166"),
    ):
        figure.add_trace(go.Scatter(x=comparison.transient.times_s, y=values, name=name,
                                    mode="lines", line={"color": color, "width": 2.3}))
    figure.add_vline(x=float(comparison.transient.times_s[snapshot]), line_color=MUTED, line_dash="dot")
    _style(figure, 315)
    figure.update_layout(xaxis_title="t [s]", yaxis_title="ΔT [K]",
                         legend={"orientation": "h", "y": 1.05, "yanchor": "bottom"},
                         margin={"l": 51, "r": 12, "t": 41, "b": 43})
    return figure


def convergence_plot(spatial: list[dict], temporal: list[dict]) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, horizontal_spacing=0.17)
    for column, study, key, label, color in (
        (1, spatial, "dx_m", "Grid spacing [m] (شبکه)", "#22D3EE"),
        (2, temporal, "dt_s", "Time step [s] (گام زمانی)", "#AC8BFF"),
    ):
        figure.add_trace(go.Scatter(
            x=[row[key] for row in study], y=[row["max_temperature_error_k"] for row in study],
            mode="lines+markers", line={"color": color, "width": 2.3}, showlegend=False,
            hovertemplate="%{x:.4g}<br>Error %{y:.4g} K<extra></extra>",
        ), row=1, col=column)
        if "pair_max_delta_k" in study[-1]:
            figure.add_trace(go.Scatter(
                x=[row[key] for row in study[1:]], y=[row["pair_max_delta_k"] for row in study[1:]],
                mode="lines+markers", line={"color": "#FFD166", "width": 1.5, "dash": "dash"},
                name="Successive change", showlegend=False,
                hovertemplate="%{x:.4g}<br>Paired change %{y:.4g} K<extra></extra>",
            ), row=1, col=column)
        figure.update_xaxes(type="log", title_text=label, row=1, col=column)
        figure.update_yaxes(type="log", title_text="Max error [K]", row=1, col=column)
    _style(figure, 300)
    return figure


def numerical_error_plot(grid: Grid, error_k: np.ndarray) -> go.Figure:
    bound = float(np.max(np.abs(error_k))) or 0.5
    figure = go.Figure(go.Heatmap(
        x0=grid.dx / 2, dx=grid.dx, y0=grid.dy / 2, dy=grid.dy, z=error_k,
        zmin=-bound, zmax=bound, colorscale="RdBu_r", zsmooth=False,
        colorbar={"title": "Error [K]", "thickness": 12, "outlinewidth": 0},
        hovertemplate="x %{x:.4g} m · y %{y:.4g} m<br>e %{z:+.4g} K<extra></extra>",
    ))
    _style(figure, 270)
    figure.update_xaxes(title="x [m]", range=[0, grid.width_m], constrain="domain")
    figure.update_yaxes(title="y [m]", range=[0, grid.height_m],
                        scaleanchor="x", constrain="domain")
    return figure
