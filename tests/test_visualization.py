"""Verify scientific display data and render real solver fields into media."""

import csv
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from heatlab.boundaries import Boundaries, Boundary, Kind
from heatlab.geometry import Grid
from heatlab.materials import Material
from heatlab.media import MapStyle, mp4_available, render_animation, render_png
from heatlab.plots import comparison_history, temperature_difference, temperature_map
from heatlab.report import comparison_csv, energy_csv, probe_csv
from heatlab.solver import solve_steady, solve_transient
from heatlab.validation import IncompatibleSteadyError
from heatlab.visualization import (
    TemperatureScale,
    compare_results,
    energy_history,
    playback_frame,
    sampled_frames,
    solve_comparison,
)

INSULATED = Boundary(Kind.INSULATED)


@pytest.fixture
def result():
    grid = Grid(0.2, 0.1, 0.02, 8, 4)
    material = Material(12, 1200, 300)
    boundaries = Boundaries(Boundary(Kind.TEMPERATURE, 350),
                            Boundary(Kind.CONVECTION, 295, 45), INSULATED, INSULATED)
    return solve_transient(grid, material, boundaries, initial_k=290, generation_w_m3=9000,
                           duration_s=120, steps=8)


def test_scale_and_isotherms_are_fixed_across_frames(result):
    scale = TemperatureScale.from_results(result)
    early = temperature_map(result, 1, unit="°C", scale=scale, contours=True)
    late = temperature_map(result, 8, unit="°C", scale=scale, contours=True)
    assert (early.data[0].zmin, early.data[0].zmax) == (late.data[0].zmin, late.data[0].zmax)
    assert early.data[1].contours.start == late.data[1].contours.start
    assert early.data[1].contours.end == late.data[1].contours.end
    assert early.data[1].contours.size == late.data[1].contours.size
    np.testing.assert_allclose(late.data[0].z, result.temperature_k[8] - 273.15)


def test_energy_integration_and_exports(result):
    energy = energy_history(result)
    assert energy.change_j[0] == 0
    assert energy.generated_j[-1] == pytest.approx(result.generated_w * 120)
    assert energy.boundary_j[-1] == pytest.approx(15 * result.inward_w[1:].sum())
    np.testing.assert_allclose(energy.change_j, energy.supplied_j, atol=1e-8, rtol=0)
    rows = list(csv.DictReader(energy_csv(result).splitlines()))
    assert float(rows[-1]["stored_change_j"]) == energy.change_j[-1]
    probes = list(csv.DictReader(probe_csv(result, ((0, 1), (2, 3))).splitlines()))
    assert len(probes) == 2 * len(result.times_s)
    assert probes[-1]["probe"] == "P2"
    assert float(probes[-1]["temperature_k"]) == result.temperature_k[-1, 2, 3]


def test_comparison_uses_same_model_and_real_difference(result):
    comparison = solve_comparison(result)
    assert comparison.transient is result
    assert comparison.steady.grid == result.grid
    assert comparison.steady.generated_w == pytest.approx(result.generated_w)
    difference = result.temperature_k[-1] - comparison.steady.temperature_k[0]
    np.testing.assert_array_equal(temperature_difference(comparison, 8).data[0].z, difference)
    assert comparison_history(comparison, 8).data[0].y[-1] == pytest.approx(np.sqrt(np.mean(difference**2)))
    assert comparison.rms_k[-1] < comparison.rms_k[0]
    rows = list(csv.DictReader(comparison_csv(comparison).splitlines()))
    assert float(rows[-1]["max_abs_delta_k"]) == pytest.approx(np.max(np.abs(difference)))
    steady_start = solve_comparison(comparison.steady, initial_k=305, duration_s=10, steps=2)
    assert steady_start.steady is comparison.steady
    assert steady_start.transient.temperature_k[0].mean() == pytest.approx(305)
    with pytest.raises(ValueError, match="steady result"):
        compare_results(result, comparison.steady)


def test_flux_only_steady_reference_and_incompatibility(result):
    adiabatic = Boundaries(INSULATED, INSULATED, INSULATED, INSULATED)
    initial = 290 + np.arange(result.grid.size).reshape(result.grid.shape)
    transient = solve_transient(result.grid, result.material, adiabatic, initial_k=initial,
                                duration_s=100, steps=5)
    comparison = solve_comparison(transient)
    assert comparison.steady.reference_mean_k == pytest.approx(initial.mean())
    assert comparison.steady.temperature_k[0].mean() == pytest.approx(initial.mean())
    transient = solve_transient(result.grid, result.material, adiabatic, initial_k=initial,
                                duration_s=100, steps=5, generation_w_m3=9000)
    with pytest.raises(IncompatibleSteadyError):
        solve_comparison(transient)


def test_playback_uses_wall_time_and_keeps_stored_indices():
    assert playback_frame(24, 120, 1.5, 1.0) == 34
    assert playback_frame(24, 120, 1.5, 2.0) == 44
    assert playback_frame(24, 120, 90, 4.0) == 120
    assert sampled_frames(121, 5) == (0, 30, 60, 90, 120)
    assert sampled_frames(3, 10) == (0, 1, 2)


def test_real_png_and_gif_exports(result):
    style = MapStyle(unit="°C", probes=((1, 1), (2, 3)))
    first = render_png(result, 0, style)
    last = render_png(result, 8, style)
    assert first != last
    with Image.open(BytesIO(last)) as png:
        assert png.format == "PNG"
        assert png.size == (1280, 800)
        assert np.asarray(png).std() > 10
    gif_bytes = render_animation(result, style, limit=4, fps=10)
    with Image.open(BytesIO(gif_bytes)) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 4
        assert gif.info["duration"] == 100
        initial_pixels = np.asarray(gif.convert("RGB"))
        gif.seek(3)
        final_pixels = np.asarray(gif.convert("RGB"))
        assert np.any(final_pixels != initial_pixels)
        np.testing.assert_array_equal(final_pixels[120:370, 708:720],
                                      initial_pixels[120:370, 708:720])
    steady = solve_steady(result.grid, result.material, result.boundaries)
    with pytest.raises(ValueError, match="transient"):
        render_animation(steady)


def test_optional_mp4_has_expected_frames(result, tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    if not mp4_available():
        pytest.skip("FFmpeg encoder is unavailable")
    path = tmp_path / "heatlab.mp4"
    path.write_bytes(render_animation(result, kind="mp4", limit=3, fps=5))
    frames, duration = ffmpeg.count_frames_and_secs(str(path))
    assert frames == 3
    assert duration == pytest.approx(0.6, abs=0.03)
