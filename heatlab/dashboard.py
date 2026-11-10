"""Streamlit result panels, animation, probes, and verification views."""

import logging
from time import monotonic

import numpy as np
import streamlit as st

from .boundaries import Side
from .media import MAX_MEDIA_FRAMES, MapStyle, mp4_available, render_animation, render_png
from .plots import (
    PLOT_CONFIG,
    cell_flux,
    comparison_history,
    convergence_plot,
    default_probes,
    energy_change_plot,
    heat_rate_history,
    nearest_index,
    numerical_error_plot,
    temperature_difference,
    temperature_history,
    temperature_map,
    temperature_surface,
)
from .report import comparison_csv, energy_csv, face_flux_csv, field_csv, history_csv, probe_csv, report_json
from .solver import Result
from .validation import HeatLabError
from .scientific_reporting import REPORT_NAME, build_report_files, report_zip
from .scientific_validation import audit_energy, run_validation
from .visualization import (
    PROBE_COLORS,
    TemperatureScale,
    energy_history,
    playback_frame,
    solve_comparison,
    temperature_values,
)

LOGGER = logging.getLogger(__name__)
INTERVAL_S = 0.25
SPEEDS = (0.5, 1.0, 2.0, 4.0)
LAYER_OPTIONS = (
    ("layer_heatmap", "Heatmap (نقشهٔ دما)"),
    ("layer_contours", "Isotherms (خطوط هم‌دما)"),
    ("layer_flux", "Heat flux (بردار شار)"),
    ("layer_surface", "3D surface (نمای سه‌بعدی)"),
)


def heading(english: str, persian: str, eyebrow: str = "") -> None:
    kicker = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.html(f'{kicker}<h2 class="panel-title">{english}'
            f'<span lang="fa" dir="rtl">{persian}</span></h2>')


def reset_view(result: Result) -> None:
    state = st.session_state
    state.run_id = state.get("run_id", 0) + 1
    state.frame = 0 if result.times_s is None else int(np.argmin(abs(result.times_s - 120)))
    state.playing = False
    state.probes = default_probes(result.grid)
    state.temperature_scale = TemperatureScale.from_results(result)
    for key in ("snapshot", "_probe_x", "_probe_y", "comparison", "comparison_frame",
                "compare_initial", "media"):
        state.pop(key, None)


def _set_frame(frame: int) -> None:
    st.session_state.frame = frame
    st.session_state.snapshot = frame
    st.session_state.playing = False


def _toggle_play(last: int) -> None:
    state = st.session_state
    state.pop("snapshot", None)
    if state.playing:
        state.playing = False
    else:
        if state.frame == last:
            state.frame = 0
        state.playing = True
        state.playback_start = state.frame
        state.playback_clock = monotonic()


def _set_speed() -> None:
    state = st.session_state
    state.playback_speed = state.speed_choice
    state.playback_start = state.frame
    state.playback_clock = monotonic()


def _time_controls(result: Result) -> int:
    state = st.session_state
    if result.times_s is None:
        st.html('<span class="time-chip">STEADY · حالت پایا</span>')
        return 0
    last = len(result.times_s) - 1
    frame = state.frame
    with st.container(key="timeline"):
        timeline, buttons = st.columns([3.3, 2.15], vertical_alignment="center", gap="small")
        with timeline:
            st.html(f'<span class="time-chip"><span class="live-dot"></span>'
                    f' t = {result.times_s[frame]:.4g} s <span class="time-count">'
                    f'/ {result.times_s[-1]:.4g} s</span></span>')
            if state.playing:
                # Auto-rerun messages may carry an older widget value. Keep the
                # read-only playhead separate from the editable slider.
                st.progress(frame / last)
            else:
                if "snapshot" not in state:
                    state.snapshot = frame
                st.slider("Time level (تراز زمانی)", 0, last, key="snapshot",
                          on_change=lambda: _set_frame(state.snapshot), label_visibility="collapsed")
        with buttons:
            with st.container(horizontal=True, gap="xxsmall", key="playback-buttons"):
                st.button("Reset", icon=":material/first_page:", key="rewind",
                          on_click=_set_frame, args=(0,), disabled=frame == 0,
                          help="First frame (فریم اول)", shortcut=None)
                st.button("Previous", icon=":material/chevron_left:", key="prev_frame",
                          on_click=_set_frame, args=(max(frame - 1, 0),), disabled=frame == 0,
                          help="Previous frame (فریم قبلی)")
                playing = state.playing
                st.button("Pause" if playing else "Play",
                          icon=":material/pause:" if playing else ":material/play_arrow:",
                          key="play", type="primary", help="Play or pause (پخش یا مکث)",
                          on_click=_toggle_play, args=(last,))
                st.button("Next", icon=":material/chevron_right:", key="next_frame",
                          on_click=_set_frame, args=(min(frame + 1, last),),
                          disabled=frame == last, help="Next frame (فریم بعدی)")
                st.selectbox("Speed (سرعت پخش)", SPEEDS,
                             index=SPEEDS.index(state.get("playback_speed", 1.0)),
                             format_func=lambda speed: f"{speed:g}×", key="speed_choice",
                             label_visibility="collapsed", width=68, on_change=_set_speed)
    return frame


def _update_probe_widgets(result: Result) -> None:
    state = st.session_state
    j, i = state.probes[state.get("active_probe", 0)]
    state._probe_x = float(result.grid.x_m[i])
    state._probe_y = float(result.grid.y_m[j])


def _set_probe(result: Result, cell: tuple[int, int]) -> None:
    state = st.session_state
    probes = list(state.probes)
    probes[state.get("active_probe", 0)] = cell
    state.probes = tuple(probes)
    _update_probe_widgets(result)


def _probe_from_coordinates(result: Result) -> None:
    state = st.session_state
    grid = result.grid
    _set_probe(result, (nearest_index(state._probe_y, grid.dy, grid.ny),
                        nearest_index(state._probe_x, grid.dx, grid.nx)))


def _probe_from_map(result: Result, key: str, curve: int) -> None:
    selected = st.session_state[key].selection.points
    if selected and selected[-1].get("curve_number") == curve:
        index = selected[-1].get("point_number")
        if isinstance(index, int) and 0 <= index < result.grid.size:
            _set_probe(result, divmod(index, result.grid.nx))


def _probe_panel(result: Result, snapshot: int, unit: str) -> None:
    heading("Probes", "نقاط اندازه‌گیری", "SENSORS")
    state = st.session_state
    st.selectbox("Active point (نقطهٔ فعال)", [0, 1, 2], format_func=lambda i: f"P{i + 1}",
                 key="active_probe", on_change=_update_probe_widgets, args=(result,))
    if "_probe_x" not in state or "_probe_y" not in state:
        _update_probe_widgets(result)
    x_col, y_col = st.columns(2, gap="small")
    x_col.number_input("x [m]", min_value=0.0, max_value=float(result.grid.width_m),
                       step=float(result.grid.dx), format="%.5f",
                       key="_probe_x", on_change=_probe_from_coordinates, args=(result,))
    y_col.number_input("y [m]", min_value=0.0, max_value=float(result.grid.height_m),
                       step=float(result.grid.dy), format="%.5f",
                       key="_probe_y", on_change=_probe_from_coordinates, args=(result,))
    st.toggle("Pick on map (انتخاب روی نقشه)", key="pick_probe", disabled=state.playing)
    rows = []
    for n, (j, i) in enumerate(state.probes):
        value = temperature_values(result.temperature_k[snapshot, j, i], unit)
        rows.append(f'<div class="probe-row"><span class="probe-dot" style="--dot:{PROBE_COLORS[n]}">'
                    f'</span><b>P{n + 1}</b><span>{result.grid.x_m[i]:.3f}, '
                    f'{result.grid.y_m[j]:.3f} m</span><strong>{value:.2f} {unit}</strong></div>')
    st.html('<div class="probe-list">' + "".join(rows) + '</div>')


def _setting(key: str, default):
    return st.session_state.get("visual_preferences", {}).get(key, default)


def _save_setting(key: str) -> None:
    state = st.session_state
    state.visual_preferences = {**state.get("visual_preferences", {}), key: state[key]}


def _map_style() -> MapStyle:
    return MapStyle(unit=_setting("display_unit", "°C"), probes=st.session_state.probes,
                    heatmap=_setting("layer_heatmap", True), isotherms=_setting("layer_contours", True),
                    flux=_setting("layer_flux", True))


def _layers() -> None:
    heading("Layers", "لایه‌های نمایش", "DISPLAY")
    for key, label in LAYER_OPTIONS:
        st.toggle(label, value=_setting(key, True), key=key, on_change=_save_setting, args=(key,))
    units = ("°C", "K")
    st.selectbox("Display unit (واحد نمایش)", units, index=units.index(_setting("display_unit", "°C")),
                 key="display_unit", on_change=_save_setting, args=("display_unit",))
    st.caption("Inputs and CSV use K. (ورودی‌ها و CSV بر حسب K هستند.)")


def _key_results(result: Result, snapshot: int, unit: str, peak_flux: float) -> None:
    heading("Key results", "نتایج کلیدی", "AT CURRENT FRAME")
    field = temperature_values(result.temperature_k[snapshot], unit)
    left, right = st.columns(2, gap="small")
    left.metric("Maximum (بیشینه)", f"{field.max():.2f} {unit}")
    right.metric("Minimum (کمینه)", f"{field.min():.2f} {unit}")
    left.metric("Mean (میانگین)", f"{field.mean():.2f} {unit}")
    right.metric("Peak |q| [W/m²] (بیشینهٔ شار)", f"{peak_flux:.3g}",
                 help="Peak cell-centred magnitude in W/m² (بیشینهٔ بزرگی شار مرکز سلول)")
    st.html('<div class="key-rate"><span>Left-side input (توان ورودی چپ)</span>'
            f'<strong>{result.inward_w[snapshot, 0]:.4g} W</strong></div>')


def _simulation_panels(result: Result, snapshot: int) -> None:
    state = st.session_state
    style = _map_style()
    unit, probes = style.unit, style.probes
    field = result.temperature_k[snapshot]
    flux = result.face_flux(snapshot)
    qx, qy = cell_flux(flux)
    peak = float(np.hypot(qx, qy).max())
    picking = state.get("pick_probe", False) and not state.playing
    if field.min() < 0:
        st.warning("Temperature fell below 0 K. Review the applied loads. "
                   "(دما از صفر کلوین کمتر شد. بار حرارتی را بررسی کنید.)")
    with st.container(key="result-top"):
        main, side = st.columns([2.25, 1], gap="small")
        with main, st.container(key="map-panel"):
            heading("Temperature field", "میدان دمای دوبعدی", "01  /  FIELD VIEW")
            snapshot = _time_controls(result)
            figure = temperature_map(
                result, snapshot, unit=unit, probes=probes, picking=picking, revision=state.run_id,
                heatmap=style.heatmap, contours=style.isotherms, flux=flux if style.flux else None,
                scale=state.temperature_scale,
            )
            options = {"theme": None, "config": PLOT_CONFIG, "key": "field_plot",
                       "alt": "Computed temperature map (نقشهٔ دمای محاسبه‌شده)"}
            if picking:
                key = f"pick_field_{state.run_id}"
                curve = len(figure.data) - 1
                options.update(key=key, selection_mode="points",
                               on_select=lambda: _probe_from_map(result, key, curve))
            st.plotly_chart(figure, **options)
            st.html('<div class="map-foot"><span><i></i> Fixed temperature scale</span>'
                    '<span lang="fa" dir="rtl">مقیاس رنگ دما در تمام فریم‌ها ثابت است</span></div>')
        with side:
            with st.container(key="metrics-panel"):
                _key_results(result, snapshot, unit, peak)
            if _setting("layer_surface", True):
                with st.container(key="surface-panel"):
                    heading("3D surface", "سطح دما", "02  /  SURFACE")
                    st.plotly_chart(temperature_surface(result, snapshot, unit, state.run_id,
                                                        scale=state.temperature_scale),
                                    theme=None, config=PLOT_CONFIG, key="surface_plot",
                                    alt="3D temperature surface (سطح سه‌بعدی دما)")
    with st.container(key="result-bottom"):
        history, points, layers = st.columns([2.15, 1.04, 0.98], gap="small")
        with history, st.container(key="history-panel"):
            heading("Temperature history", "تاریخچهٔ دما", "03  /  PROBE TRACES")
            if result.times_s is None:
                st.info("Time traces are available in transient mode. (نمودار زمانی در حالت ناپایا فعال است.)")
            else:
                st.plotly_chart(temperature_history(result, probes, snapshot, unit),
                                theme=None, config=PLOT_CONFIG, key="history_plot",
                                alt="Temperature histories at the probes (تاریخچهٔ دمای نقاط)")
        with points, st.container(key="probe-panel"):
            _probe_panel(result, snapshot, unit)
        with layers, st.container(key="layers-panel"):
            _layers()


def simulation_view(result: Result) -> None:
    scheduled = st.session_state.playing

    @st.fragment(run_every=INTERVAL_S if scheduled else None)
    def render() -> None:
        state = st.session_state
        if scheduled != state.playing:
            st.rerun()
        if scheduled:
            last = len(result.times_s) - 1
            frame = playback_frame(state.playback_start, last, monotonic() - state.playback_clock,
                                   state.get("playback_speed", 1.0))
            state.frame = frame
            if frame == last:
                state.playing = False
                state.pop("snapshot", None)
                st.rerun()
        _simulation_panels(result, state.frame)

    render()


def _energy_panel(result: Result, frame: int) -> None:
    inward = result.inward_w[frame]
    cols = st.columns(3 if result.times_s is None else 4, gap="small")
    cols[0].metric("Net boundary input (ورودی مرزی)", f"{inward.sum():.5g} W")
    cols[1].metric("Heat generation (تولید حجمی)", f"{result.generated_w:.5g} W")
    cols[2].metric("Stored energy (انرژی ذخیره‌شده)", f"{result.stored_energy_j[frame]:.5g} J")
    if result.times_s is not None:
        energy = energy_history(result)
        cols[3].metric("Stored ΔE (تغییر انرژی ذخیره‌شده)", f"{energy.change_j[frame]:.5g} J")
        st.plotly_chart(energy_change_plot(result, frame), theme=None, config=PLOT_CONFIG,
                        key="energy_change", alt="Cumulative supplied and stored energy (انرژی تجمعی)")
        st.caption(f"Cumulative residual (باقیماندهٔ تجمعی) · {energy.residual_j[frame]:.4g} J")
        with st.expander("Instantaneous heat rates (توان لحظه‌ای)"):
            st.plotly_chart(heat_rate_history(result, frame), theme=None, config=PLOT_CONFIG,
                            key="rate_history")
    st.table({"Side (ضلع)": [side.value for side in Side], "Inward rate [W] (توان ورودی)": inward})


def _comparison_panel(result: Result) -> None:
    state = st.session_state
    heading("Steady vs transient", "مقایسهٔ پایا و ناپایا")
    st.caption("Both solutions use the same grid, material and loads. "
               "(هر دو حل از شبکه، ماده و بار یکسان استفاده می‌کنند.)")
    settings = {}
    if result.times_s is None:
        a, b, c = st.columns(3, gap="small")
        reference = result.reference_mean_k
        settings = {
            "initial_k": a.number_input("Initial T [K] (دمای اولیه)", min_value=0.0,
                                        value=298.15 if reference is None else reference,
                                        key="compare_initial"),
            "duration_s": b.number_input("Duration [s] (مدت)", min_value=0.0,
                                         value=600.0, key="compare_duration"),
            "steps": c.number_input("Time steps (گام‌ها)", 1, 2000, 120, key="compare_steps"),
        }
    elif not result.boundaries.has_temperature_anchor:
        st.caption("For compatible flux-only loads, the steady mean uses the initial mean. "
                   "(برای بار شارمحور سازگار، میانگین اولیه مرجع حل پایاست.)")
    spec = (state.run_id, tuple(settings.values()))
    if st.button("Solve comparison (حل برای مقایسه)", icon=":material/compare:",
                 type="primary", key="compare_solve"):
        state.pop("comparison", None)
        state.pop("comparison_frame", None)
        try:
            with st.spinner("Solving the companion regime (در حال حل حالت دوم)"):
                state.comparison = (spec, solve_comparison(result, **settings), None)
        except HeatLabError as exc:
            state.comparison = (spec, None, str(exc))
    saved = state.get("comparison")
    if saved is None or saved[0] != spec:
        return
    _, comparison, error = saved
    if error:
        st.error(error)
        return
    times = comparison.transient.times_s
    frame = st.slider("Comparison frame (تراز مقایسه)", 0, len(times) - 1,
                      min(state.frame, len(times) - 1) if result.times_s is not None else len(times) - 1,
                      key="comparison_frame")
    unit = _setting("display_unit", "°C")
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**Steady (پایا)**")
        st.plotly_chart(temperature_map(comparison.steady, 0, unit=unit, scale=comparison.scale, height=315),
                        theme=None, config=PLOT_CONFIG, key="comparison_steady")
    with right:
        st.markdown(f"**Transient (ناپایا) · {times[frame]:.4g} s**")
        st.plotly_chart(temperature_map(comparison.transient, frame, unit=unit, scale=comparison.scale,
                                        height=315),
                        theme=None, config=PLOT_CONFIG, key="comparison_transient")
    st.caption("The two temperature maps share one fixed color scale. "
               "(دو نقشهٔ دما یک مقیاس رنگ ثابت و مشترک دارند.)")
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**Transient − steady (اختلاف دما)**")
        st.plotly_chart(temperature_difference(comparison, frame), theme=None, config=PLOT_CONFIG,
                        key="comparison_difference")
    with right:
        st.markdown("**Difference over time (تغییرات اختلاف)**")
        st.plotly_chart(comparison_history(comparison, frame), theme=None, config=PLOT_CONFIG,
                        key="comparison_history")
    st.download_button("Comparison CSV (مقایسه)", comparison_csv(comparison),
                       "heatlab-comparison.csv", "text/csv", on_click="ignore")


def _export_panel(result: Result, frame: int) -> None:
    state = st.session_state
    with st.container(horizontal=True, gap="small"):
        st.download_button("Field CSV (میدان)", field_csv(result, frame),
                           f"heatlab-field-{frame}.csv", "text/csv", on_click="ignore")
        st.download_button("Flux CSV (شار)", face_flux_csv(result, frame),
                           f"heatlab-flux-{frame}.csv", "text/csv", on_click="ignore")
        st.download_button("Report JSON (گزارش)", report_json(result, frame),
                           f"heatlab-report-{frame}.json", "application/json", on_click="ignore")
        if result.times_s is not None:
            for label, data, name in (
                ("History CSV (تاریخچه)", history_csv(result), "history"),
                ("Probes CSV (نقاط)", probe_csv(result, state.probes), "probes"),
                ("Energy CSV (انرژی)", energy_csv(result), "energy"),
            ):
                st.download_button(label, data, f"heatlab-{name}.csv", "text/csv", on_click="ignore")

    heading("Image and animation", "تصویر و انیمیشن")
    kinds = ["png"]
    if result.times_s is not None:
        kinds.append("gif")
        if mp4_available():
            kinds.append("mp4")
    controls = st.columns(3, gap="small")
    kind = controls[0].selectbox("Format (قالب)", kinds, format_func=str.upper, key="media_kind")
    limit, fps = 20, 10
    if kind != "png":
        maximum = min(len(result.times_s), MAX_MEDIA_FRAMES)
        limit = controls[1].number_input("Frames (تعداد فریم)", 2, maximum, min(20, maximum),
                                         key="media_frames") if maximum > 2 else 2
        fps = controls[2].selectbox("Frames per second (فریم بر ثانیه)", [5, 10, 20], index=1,
                                    key="media_fps")
        st.caption("Animation samples stored frames from first to last. "
                   "(انیمیشن از ترازهای ذخیره‌شدهٔ ابتدا تا انتهای حل نمونه می‌گیرد.)")
    if result.times_s is not None and "mp4" not in kinds:
        st.caption("Optional MP4 support (پشتیبانی اختیاری MP4) · requirements-video.txt")
    style = _map_style()
    spec = (state.run_id, frame if kind == "png" else None, style, kind, limit, fps)
    if st.button(f"Prepare {kind.upper()} (آماده‌سازی)", icon=":material/image:",
                 type="primary", key=f"prepare_{kind}"):
        state.pop("media", None)
        try:
            with st.spinner("Rendering solved fields (در حال ترسیم میدان‌های حل‌شده)"):
                data = (render_png(result, frame, style) if kind == "png" else
                        render_animation(result, style, kind=kind, limit=limit, fps=fps))
                state.media = (spec, data, None)
        except (ImportError, OSError, RuntimeError, ValueError):
            LOGGER.exception("Media export failed")
            state.media = (spec, None, "Export failed. Check the application log. "
                           "(ساخت فایل انجام نشد. گزارش برنامه را بررسی کنید.)")
    saved = state.get("media")
    if saved is not None and saved[0] == spec:
        _, data, error = saved
        if error:
            st.error(error)
        else:
            name = f"heatlab-frame-{frame}.png" if kind == "png" else f"heatlab-temperature.{kind}"
            mime = "video/mp4" if kind == "mp4" else f"image/{kind}"
            st.download_button(f"Download {kind.upper()} (دریافت)", data, name, mime,
                               on_click="ignore", key=f"download_{kind}")


def analysis_view(result: Result) -> None:
    with st.container(key="analysis-panel"):
        heading("Analysis", "تحلیل نتایج", "02  /  WORKSPACE")
        views = ("Energy (انرژی)", "Compare (مقایسه)", "Export (خروجی)")
        view = st.segmented_control("Analysis view (نمای تحلیل)", views, default=views[0],
                                    required=True, key="analysis_view", label_visibility="collapsed")
        if view == views[1]:
            _comparison_panel(result)
            return
        frame = st.session_state.frame
        if result.times_s is not None:
            if "analysis_frame" not in st.session_state:
                st.session_state.analysis_frame = frame
            frame = st.slider("Result frame (تراز زمانی نتیجه)", 0, len(result.times_s) - 1,
                              key="analysis_frame")
            st.caption(f"t = {result.times_s[frame]:.4g} s")
        if view == views[2]:
            _export_panel(result, frame)
        else:
            _energy_panel(result, frame)


def validation_view(result: Result | None) -> None:
    with st.container(key="validation-panel"):
        state = st.session_state
        heading("Validation", "اعتبارسنجی", "03  /  WORKSPACE")
        if result is not None:
            audit = audit_energy(result)
            st.metric("Current run · max |power residual| (بیشینهٔ باقیماندهٔ توان حل فعلی)",
                      f"{audit.max_abs_power_w:.4g} W")
        st.caption("The benchmarks below solve independent reference problems. "
                   "(آزمون‌های زیر مسائل مرجع مستقل را حل می‌کنند.)")
        if st.button("Run benchmarks (اجرای آزمون‌های مرجع)", icon=":material/check_circle:",
                     type="primary", key="benchmark"):
            with st.spinner("Solving reference problems (در حال حل مسائل مرجع)"):
                state.validation_run = run_validation()
                state.pop("validation_report", None)
        run = state.get("validation_run")
        if run is None:
            return
        cols = st.columns(3, gap="small")
        cols[0].metric("Spatial order (مرتبهٔ مکانی)", f"{run.spatial[-1]['order']:.3f}")
        cols[1].metric("Temporal order (مرتبهٔ زمانی)", f"{run.temporal[-1]['order']:.3f}")
        cols[2].metric("Max. energy residual (بیشینهٔ باقیماندهٔ انرژی)",
                       f"{run.transient_energy.max_abs_energy_j:.3g} J")
        st.plotly_chart(convergence_plot(run.spatial, run.temporal), theme=None, config=PLOT_CONFIG,
                        key="convergence_plot",
                        alt="Spatial and temporal convergence on log scales (همگرایی مکانی و زمانی)")
        with st.expander("Signed error fields (میدان خطای علامت‌دار)"):
            steady_col, transient_col = st.columns(2, gap="small")
            with steady_col:
                st.caption("Steady vs analytical (پایا در برابر تحلیلی)")
                st.plotly_chart(numerical_error_plot(run.steady.grid,
                                                    run.steady.temperature_k[0] - run.steady_exact),
                                theme=None, config=PLOT_CONFIG, key="steady_error")
            with transient_col:
                st.caption("Transient vs continuum (ناپایا در برابر تحلیلی پیوسته)")
                st.plotly_chart(numerical_error_plot(run.transient.grid,
                                                    run.transient.temperature_k[-1] - run.transient_continuum),
                                theme=None, config=PLOT_CONFIG, key="transient_error")
        with st.expander("Grid errors and independence (خطا و استقلال شبکه)", expanded=True):
            st.dataframe(run.spatial, hide_index=True)
        with st.expander("Time errors and independence (خطا و استقلال زمانی)"):
            st.dataframe(run.temporal, hide_index=True)
        heading("Scientific report", "گزارش علمی")
        include = result is not None and st.checkbox(
            "Include the current simulation appendix (افزودن پیوست شبیه‌سازی فعلی)", value=True,
            key="validation_include_current",
        )
        token = state.run_id if include else None
        if st.button("Prepare PDF and data (آماده‌سازی PDF و داده‌ها)", key="validation_prepare",
                     icon=":material/description:", type="primary"):
            state.pop("validation_report", None)
            try:
                with st.spinner("Rendering report (در حال ساخت گزارش)"):
                    files = build_report_files(run, result if include else None)
                    state.validation_report = (token, files[REPORT_NAME], report_zip(files))
            except (ImportError, OSError, RuntimeError, ValueError):
                LOGGER.exception("Scientific report export failed")
                st.error("Report export failed. Check the application log. "
                         "(ساخت گزارش انجام نشد. گزارش برنامه را بررسی کنید.)")
        saved = state.get("validation_report")
        if saved is not None and saved[0] == token:
            with st.container(horizontal=True, gap="small"):
                st.download_button("Report PDF (گزارش PDF)", saved[1], REPORT_NAME,
                                   "application/pdf", on_click="ignore", key="validation_pdf")
                st.download_button("CSV & PNG package (بستهٔ داده و نمودار)", saved[2],
                                   "HeatLab2D-Validation-Data.zip", "application/zip",
                                   on_click="ignore", key="validation_zip")
