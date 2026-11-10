"""Browser-independent PNG and GIF export, with optional H.264/MP4."""

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np
from PIL import Image

from .plots import flux_vectors
from .solver import Result
from .visualization import PROBE_COLORS, THERMAL, Cell, TemperatureScale, sampled_frames, temperature_values

MAX_MEDIA_FRAMES = 30
RENDER_LOCK = Lock()  # Matplotlib's shared font and artist caches are not thread-safe.


@dataclass(frozen=True)
class MapStyle:
    unit: str = "K"
    probes: tuple[Cell, ...] = ()
    heatmap: bool = True
    isotherms: bool = True
    flux: bool = True


class _MapCanvas:
    def __init__(self, result: Result, style: MapStyle, *, dpi: int) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.figure import Figure

        self.result = result
        self.style = style
        self.scale = TemperatureScale.from_results(result)
        self.levels = self.scale.levels(style.unit)
        grid = result.grid
        ink, text = "#0B1120", "#E9F1FF"
        figure = Figure(figsize=(8, 5), dpi=dpi, facecolor=ink)
        self.canvas = FigureCanvasAgg(figure)
        self.axis = figure.add_axes((0.11, 0.15, 0.74, 0.70), facecolor="#111d31")
        self.axis.set(xlabel="x [m]", ylabel="y [m]", xlim=(0, grid.width_m),
                      ylim=(0, grid.height_m), aspect="equal")
        self.axis.tick_params(colors=text, labelsize=9)
        self.axis.xaxis.label.set_color(text)
        self.axis.yaxis.label.set_color(text)
        for spine in self.axis.spines.values():
            spine.set_color("#587193")
        figure.text(0.11, 0.925, "HeatLab 2D   /   Temperature field",
                    color=text, weight="bold", fontsize=13)
        self.time_label = figure.text(0.89, 0.925, "", color="#22D3EE", ha="right", fontsize=10)
        self.mesh = None
        if style.heatmap:
            low, high = self.scale.limits(style.unit)
            self.mesh = self.axis.pcolormesh(
                grid.x_faces_m, grid.y_faces_m, np.zeros(grid.shape), shading="flat",
                cmap=LinearSegmentedColormap.from_list("heatlab", THERMAL),
                norm=Normalize(low, high), rasterized=True,
            )
            color_axis = figure.add_axes((0.88, 0.22, 0.022, 0.54))
            colorbar = figure.colorbar(self.mesh, cax=color_axis)
            colorbar.set_label(f"T [{style.unit}]", color=text)
            colorbar.ax.tick_params(colors=text, labelsize=9)
            colorbar.outline.set_edgecolor("#587193")
        self.overlay = []
        for number, ((j, i), color) in enumerate(zip(style.probes, PROBE_COLORS), start=1):
            x, y = grid.x_m[i], grid.y_m[j]
            self.axis.scatter(x, y, c=color, edgecolors="white", s=32, linewidths=0.8, zorder=5)
            self.axis.annotate(f"P{number}", (x, y), xytext=(0, 8), textcoords="offset points",
                               ha="center", color=text, weight="bold", fontsize=8, zorder=6)

    def render(self, snapshot: int) -> Image.Image:
        for artist in self.overlay:
            artist.remove()
        self.overlay.clear()
        result, style = self.result, self.style
        grid = result.grid
        values = temperature_values(result.temperature_k[snapshot], style.unit)
        if self.mesh is not None:
            self.mesh.set_array(values)
        if style.isotherms and min(grid.shape) > 1:
            levels = self.levels[(self.levels > values.min()) & (self.levels < values.max())]
            if levels.size:
                contours = self.axis.contour(
                    grid.x_m, grid.y_m, values, levels=levels,
                    colors="#ffffff", linewidths=0.8, alpha=0.76, zorder=3,
                )
                self.axis.clabel(contours, inline=True, fmt="%.3g", fontsize=7, zorder=6)
                self.overlay.append(contours)
        if style.flux:
            x, y, u, v = flux_vectors(grid, result.face_flux(snapshot))
            if x.size:
                self.overlay.append(self.axis.quiver(
                    x, y, u, v, angles="xy", scale_units="xy", scale=1, pivot="middle",
                    color=(1, 1, 1, 0.73), width=0.0028, zorder=4,
                ))
        self.time_label.set_text("Steady" if result.times_s is None
                                 else f"t = {result.times_s[snapshot]:.10g} s")
        self.canvas.draw()
        return Image.fromarray(np.asarray(self.canvas.buffer_rgba())).convert("RGB")


@lru_cache(maxsize=1)
def mp4_available() -> bool:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return bool(get_ffmpeg_exe())
    except (ImportError, OSError, RuntimeError):
        return False


def render_png(result: Result, snapshot: int, style: MapStyle = MapStyle()) -> bytes:
    with RENDER_LOCK:
        image = _MapCanvas(result, style, dpi=160).render(snapshot)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _gif(canvas: _MapCanvas, frames: tuple[int, ...], fps: int) -> bytes:
    # The first frame's colorbar spans the full temperature range. A common palette
    # keeps its color mapping stable across GIF frames.
    initial = canvas.render(frames[0])
    palette = initial.convert("P", palette=Image.Palette.ADAPTIVE)
    first = initial.quantize(palette=palette, dither=Image.Dither.NONE)
    others = [canvas.render(index).quantize(palette=palette, dither=Image.Dither.NONE)
              for index in frames[1:]]
    buffer = BytesIO()
    first.save(buffer, format="GIF", save_all=True, append_images=others,
               duration=round(100 / fps) * 10, loop=0, disposal=2)
    return buffer.getvalue()


def _mp4(canvas: _MapCanvas, frames: tuple[int, ...], fps: int) -> bytes:
    from imageio_ffmpeg import write_frames

    with TemporaryDirectory(prefix="heatlab-video-") as directory:
        path = Path(directory) / "temperature.mp4"
        writer = write_frames(
            str(path), canvas.canvas.get_width_height(), fps=fps, codec="libx264",
            pix_fmt_out="yuv420p", macro_block_size=1, ffmpeg_timeout=15,
            output_params=["-movflags", "+faststart"],
        )
        try:
            writer.send(None)
            for index in frames:
                writer.send(canvas.render(index).tobytes())
        finally:
            writer.close()
        data = path.read_bytes() if path.is_file() else b""
    if b"ftyp" not in data[:16] or b"moov" not in data[:4096]:
        raise RuntimeError("FFmpeg did not produce a playable MP4.")
    return data


def render_animation(result: Result, style: MapStyle = MapStyle(), *,
                     kind: str = "gif", limit: int = 20, fps: int = 10) -> bytes:
    if result.times_s is None:
        raise ValueError("An animation requires a transient result.")
    if not 2 <= limit <= MAX_MEDIA_FRAMES or not 1 <= fps <= 30:
        raise ValueError("Use 2–30 frames and 1–30 frames per second.")
    if kind not in ("gif", "mp4"):
        raise ValueError("Select GIF or MP4.")
    if kind == "mp4" and not mp4_available():
        raise RuntimeError("MP4 requires imageio-ffmpeg.")
    frames = sampled_frames(len(result.times_s), limit)
    with RENDER_LOCK:
        canvas = _MapCanvas(result, style, dpi=100)
        return _gif(canvas, frames, fps) if kind == "gif" else _mp4(canvas, frames, fps)
