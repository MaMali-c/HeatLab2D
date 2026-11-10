"""HeatLab 2D bilingual Streamlit workbench."""

import base64
import logging
from pathlib import Path

import streamlit as st

from heatlab.boundaries import Boundaries, Boundary, Kind, Side
from heatlab.dashboard import analysis_view, heading, reset_view, simulation_view, validation_view
from heatlab.geometry import Grid
from heatlab.materials import Material
from heatlab.solver import MAX_STORED_VALUES, Result, solve_steady, solve_transient
from heatlab.validation import HeatLabError

STEADY = "Steady (پایا)"
TRANSIENT = "Transient (ناپایا)"
ASSETS = Path(__file__).resolve().parent / "assets"
LOGGER = logging.getLogger(__name__)
MATERIALS = {
    "Aluminium (آلومینیوم)": (237.0, 2700.0, 897.0),
    "Copper (مس)": (401.0, 8960.0, 385.0),
    "Stainless steel (فولاد زنگ‌نزن)": (15.0, 7800.0, 500.0),
    "Custom (سفارشی)": None,
}
BC_LABELS = {
    Kind.TEMPERATURE: "Fixed T (دمای ثابت)",
    Kind.FLUX: "Flux (شار)",
    Kind.INSULATED: "Insulated (عایق)",
    Kind.CONVECTION: "Convection (همرفت)",
}
NAV = (
    ("Simulation", "شبیه‌سازی", "monitoring"),
    ("Analysis", "تحلیل", "query_stats"),
    ("Validation", "اعتبارسنجی", "verified"),
)


@st.cache_data(show_spinner=False)
def font_data() -> str:
    return base64.b64encode((ASSETS / "Vazirmatn.woff2").read_bytes()).decode("ascii")


def brand_header() -> None:
    st.html(
        '<style>@font-face{font-family:Vazirmatn;src:url(data:font/woff2;base64,'
        + font_data() + ') format("woff2");font-weight:100 900;font-display:swap}</style>'
    )
    st.html(ASSETS / "dashboard.css")
    st.html("""
        <header class="hl-header">
          <div class="hl-brand">
            <span class="hl-logo" aria-hidden="true"></span>
            <div><h1>HeatLab <em>2D</em></h1><small>THERMAL SIMULATION LAB</small></div>
          </div>
          <div class="hl-header-title">
            <strong lang="fa" dir="rtl">آزمایشگاه رسانش حرارتی دوبعدی</strong>
            <span>2D heat conduction · Steady &amp; transient</span>
          </div>
          <div class="engine-chip"><span class="live-dot"></span><span>FINITE VOLUME</span></div>
          <a class="mobile-setup" href="#simulation-setup">Setup <span lang="fa">(تنظیمات)</span></a>
        </header>
    """)


def _navigate(section: str) -> None:
    st.session_state.section = section
    st.session_state.playing = False


def navigation() -> str:
    state = st.session_state
    active = state.get("section", "Simulation")
    with st.container(key="navigation"):
        st.html('<div class="rail-label">WORKSPACE</div>')
        for name, persian, icon in NAV:
            st.button(f"{name}  \n{persian}", icon=f":material/{icon}:", key=f"nav_{name}",
                      type="primary" if name == active else "secondary", width="stretch",
                      on_click=_navigate, args=(name,))
        st.html('<div class="rail-foot"><span>SI</span> UNITS</div>')
    return active


def boundary_inputs(side: Side) -> tuple[Kind, float, float]:
    default = Kind.TEMPERATURE if side is Side.LEFT else Kind.INSULATED
    st.html(f'<div class="side-label">{side.value}</div>')
    choice, field = st.columns([1.4, 1], gap="xxsmall", vertical_alignment="top")
    with choice:
        kind = st.selectbox(f"{side.value} boundary type", list(Kind), index=list(Kind).index(default),
                            format_func=BC_LABELS.__getitem__, key=f"{side.name}_kind",
                            label_visibility="collapsed")
    with field:
        if kind is Kind.TEMPERATURE:
            value = st.number_input("Surface T [K] (دمای سطح)", min_value=0.0,
                                    value=423.15 if side is Side.LEFT else 298.15, format="%.2f",
                                    key=f"{side.name}_surface", label_visibility="collapsed")
            return kind, value, 0.0
        if kind is Kind.FLUX:
            value = st.number_input("Flux [W/m²] (شار)", value=0.0, format="%.4g",
                                    key=f"{side.name}_flux", label_visibility="collapsed")
            return kind, value, 0.0
        if kind is Kind.CONVECTION:
            ambient = st.number_input("Ambient [K] (محیط)", min_value=0.0, value=298.15,
                                      format="%.2f", key=f"{side.name}_ambient")
            h = st.number_input("h [W/m²K]", min_value=0.0, value=10.0, key=f"{side.name}_h")
            return kind, ambient, h
        st.html('<div class="adiabatic">q = 0</div>')
    return kind, 0.0, 0.0


def material_inputs() -> tuple[float, float, float]:
    name = st.selectbox("Material (ماده)", list(MATERIALS), key="material",
                        label_visibility="collapsed")
    values = MATERIALS[name]
    if values is None:
        values = (
            st.number_input("k [W/(m·K)] (رسانندگی)", min_value=0.0, value=237.0,
                            key="custom_k", format="%.6g"),
            st.number_input("ρ [kg/m³] (چگالی)", min_value=0.0, value=2700.0, key="custom_rho"),
            st.number_input("cₚ [J/(kg·K)] (گرمای ویژه)", min_value=0.0, value=897.0,
                            key="custom_cp"),
        )
    else:
        k, rho, cp = values
        st.html(f'<div class="material-readout"><span>k <b>{k:g}</b> W/(m·K)</span>'
                f'<span>ρ <b>{rho:g}</b> kg/m³</span><span>cₚ <b>{cp:g}</b> J/(kg·K)</span>'
                f'<span>α <b>{k / (rho * cp):.2e}</b> m²/s</span></div>')
    return values


def model_inputs() -> tuple:
    heading("Simulation setup", "تنظیمات حل", "MODEL")
    mode = st.segmented_control("Mode (حالت حل)", [TRANSIENT, STEADY], default=TRANSIENT,
                                required=True, key="mode", label_visibility="collapsed", width="stretch")
    with st.expander("Geometry (هندسه)", expanded=True, icon=":material/aspect_ratio:"):
        x_col, y_col = st.columns(2, gap="small")
        width = x_col.number_input("Lₓ [m] (طول)", min_value=0.0, value=0.20, format="%.6g",
                                   key="width")
        height = y_col.number_input("Lᵧ [m] (عرض)", min_value=0.0, value=0.20, format="%.6g",
                                    key="height")
        thickness = st.number_input("Thickness [m] (ضخامت)", min_value=0.0, value=0.005,
                                    format="%.6g", key="thickness")
    with st.expander("Material (ماده)", expanded=True, icon=":material/layers:"):
        properties = material_inputs()
    with st.expander("Boundary conditions (شرایط مرزی)", expanded=True, icon=":material/border_outer:"):
        st.caption("T [K] · Inward flux [W/m²] (شار ورودی)")
        conditions = tuple(boundary_inputs(side) for side in Side)
    with st.expander("Solver settings (تنظیمات حل‌گر)", expanded=True, icon=":material/tune:"):
        x_col, y_col = st.columns(2, gap="small")
        nx = x_col.number_input("Cells x (سلول)", 1, 100, 48, key="nx")
        ny = y_col.number_input("Cells y (سلول)", 1, 100, 48, key="ny")
        time = None
        mean = None
        if mode == TRANSIENT:
            initial = st.number_input("Initial T [K] (دمای اولیه)", min_value=0.0, value=298.15,
                                      key="initial", format="%.2f")
            time_col, steps_col = st.columns(2, gap="small")
            duration = time_col.number_input("Time [s] (مدت)", min_value=0.0, value=600.0,
                                             key="duration")
            steps = steps_col.number_input("Steps (گام)", 1, 2000, 120, key="steps")
            time = (initial, duration, steps)
            st.caption(f"Δt = {duration / steps:.4g} s")
            limit = min(2000, MAX_STORED_VALUES // (nx * ny) - 1)
            if steps > limit:
                st.warning(f"This grid supports at most {limit:,} time steps. "
                           f"(سقف گام‌های زمانی برای این شبکه {limit:,} است.)")
        elif all(item[0] in (Kind.FLUX, Kind.INSULATED) for item in conditions):
            mean = st.number_input("Reference mean [K] (میانگین مرجع)", min_value=0.0, value=298.15,
                                   key="reference_mean")
        generation = st.number_input("Generation [W/m³] (تولید حجمی)", value=0.0, key="generation",
                                     format="%.4g", help="Negative values remove heat (مقدار منفی گرما می‌گیرد)")
    return (width, height, thickness, nx, ny), properties, conditions, time, mean, generation


def simulate(settings: tuple) -> Result:
    geometry, properties, conditions, time, mean, generation = settings
    grid = Grid(*geometry)
    material = Material(*properties)
    boundaries = Boundaries(*(Boundary(*condition) for condition in conditions))
    if time is None:
        return solve_steady(grid, material, boundaries, reference_mean_k=mean,
                            generation_w_m3=generation)
    initial, duration, steps = time
    return solve_transient(grid, material, boundaries, initial_k=initial,
                           duration_s=duration, steps=steps, generation_w_m3=generation)


def _current_result(settings: tuple, submitted: bool) -> Result | None:
    state = st.session_state
    if submitted or not state.get("attempted", False):
        state.attempted = True
        state.playing = False
        state.pop("simulation", None)
        state.pop("simulation_error", None)
        state.pop("analysis_frame", None)
        try:
            with st.spinner("Solving (در حال حل)"):
                result = simulate(settings)
                state.simulation = settings, result
                reset_view(result)
        except HeatLabError as exc:
            state.simulation_error = str(exc)
        except Exception:
            LOGGER.exception("Unexpected simulation failure")
            state.simulation_error = ("Simulation failed. Check the application log. "
                                      "(حل ناموفق بود. گزارش برنامه را بررسی کنید.)")
    previous = state.get("simulation")
    if previous is not None and previous[0] == settings:
        return previous[1]
    state.playing = False
    if "simulation_error" in state:
        st.error(state.simulation_error)
    elif previous is not None:
        st.info("Inputs changed. Run the simulation to update the results. "
                "(ورودی‌ها تغییر کرده‌اند. شبیه‌سازی را دوباره اجرا کنید.)")
    return None


def main() -> None:
    st.set_page_config(page_title="HeatLab 2D", page_icon="🌡️", layout="wide",
                       initial_sidebar_state="collapsed")
    brand_header()
    with st.container(key="workbench"):
        rail, inputs, workspace = st.columns([0.83, 2.62, 9.0], gap="small")
        with rail:
            section = navigation()
        with inputs, st.container(key="settings-panel"):
            st.html('<div id="simulation-setup" class="settings-anchor"></div>')
            with st.container(key="setup-scroll"):
                settings = model_inputs()
            _, _, _, time, _, _ = settings
            _, _, _, nx, ny = settings[0]
            over_budget = time is not None and nx * ny * (time[2] + 1) > MAX_STORED_VALUES
            submitted = st.button("Run simulation (اجرای شبیه‌سازی)", icon=":material/play_arrow:",
                                  key="run", type="primary", width="stretch", disabled=over_budget)
        with workspace, st.container(key="main-panel"):
            result = _current_result(settings, submitted)
            if section == "Validation":
                validation_view(result)
            elif result is not None:
                if section == "Analysis":
                    analysis_view(result)
                else:
                    simulation_view(result)


if __name__ == "__main__":
    main()
