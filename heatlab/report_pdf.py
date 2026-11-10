"""University-style PDF for the HeatLab analytical verification study."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from .boundaries import Kind, Side
from .scientific_validation import ValidationRun, current_run_manifest, validation_manifest
from .solver import Result

WIDTH, HEIGHT = A4
MARGIN = 51
CONTENT = WIDTH - 2 * MARGIN
INK = colors.HexColor("#122940")
TEAL = colors.HexColor("#087F8C")
MUTED = colors.HexColor("#435B6F")
RULE = colors.HexColor("#D2DFE9")
PALE = colors.HexColor("#EFF5F8")
WILEY_TOC = "https://catalogimages.wiley.com/images/db/pdf/9781119320432.toc.pdf"


@lru_cache(maxsize=1)
def _fonts() -> None:
    from matplotlib import get_data_path

    base = Path(get_data_path()) / "fonts" / "ttf"
    for name, filename in (("Report", "DejaVuSans.ttf"), ("Report-Bold", "DejaVuSans-Bold.ttf"),
                           ("Report-Italic", "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(base / filename)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold", italic="Report-Italic",
                                  boldItalic="Report-Bold")
    font = Path(__file__).resolve().parent.parent / "assets" / "Vazirmatn-Regular.ttf"
    pdfmetrics.registerFont(TTFont("VazirmatnReport", str(font)))


def _styles() -> dict:
    body = ParagraphStyle("body", fontName="Report", fontSize=9.3, leading=14.5,
                          textColor=INK, spaceAfter=9)
    return {
        "body": body,
        "title": ParagraphStyle("title", parent=body, fontName="Report-Bold", fontSize=26,
                                leading=30.5, spaceAfter=9),
        "kicker": ParagraphStyle("kicker", parent=body, fontName="Report-Bold", fontSize=8.2,
                                 leading=11, textColor=TEAL, spaceAfter=8),
        "sub": ParagraphStyle("sub", parent=body, fontSize=10.5, leading=16, textColor=MUTED,
                              spaceAfter=15),
        "heading": ParagraphStyle("heading", parent=body, fontName="Report-Bold", fontSize=12,
                                  leading=16, spaceBefore=9, spaceAfter=5),
        "small": ParagraphStyle("small", parent=body, fontSize=8.05, leading=12,
                                textColor=MUTED, spaceAfter=7),
        "equation": ParagraphStyle("equation", parent=body, fontSize=9.3, leading=15,
                                   leftIndent=8, rightIndent=6, textColor=TEAL, spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=body, fontSize=7.65, leading=10.7, spaceAfter=0),
        "th": ParagraphStyle("th", parent=body, fontName="Report-Bold", fontSize=7.65,
                             leading=10.7, textColor=colors.white, spaceAfter=0),
        "persian": ParagraphStyle("persian", fontName="VazirmatnReport", fontSize=11,
                                  leading=16.5, alignment=TA_RIGHT, textColor=INK),
    }


def _chrome(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, HEIGHT - 8, WIDTH, 8, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(MARGIN, HEIGHT - 8, 46, 8, fill=1, stroke=0)
    canvas.setFont("Report-Bold", 8)
    canvas.setFillColor(INK)
    canvas.drawString(MARGIN, HEIGHT - 36, "HEATLAB 2D")
    canvas.setFont("Report", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 36, "COMPUTATIONAL HEAT TRANSFER")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 42, WIDTH - MARGIN, 42)
    canvas.drawString(MARGIN, 27, "Numerical verification  /  Scientific report")
    canvas.drawRightString(WIDTH - MARGIN, 27, f"{doc.page:02d}")
    canvas.restoreState()


def _boundary(result, side: Side) -> str:
    bc = result.boundaries[side]
    if bc.kind is Kind.INSULATED:
        return "Adiabatic"
    if bc.kind is Kind.TEMPERATURE:
        return f"Fixed T = {bc.value:g} K"
    if bc.kind is Kind.FLUX:
        return f"Inward flux = {bc.value:g} W/m²"
    return f"T<sub>∞</sub> = {bc.value:g} K; h = {bc.h_w_m2k:g} W/(m² K)"


def build_pdf(run: ValidationRun, figures: dict[str, bytes],
              current: Result | None = None) -> bytes:
    _fonts()
    styles = _styles()
    manifest = validation_manifest(run)
    spatial, temporal = run.spatial[-1], run.temporal[-1]
    audit_a = manifest["energy_audit"]["steady"]
    audit_c = manifest["energy_audit"]["transient"]
    acceptance = manifest["acceptance"]
    verdict = lambda passed: "Within threshold" if passed else "Above threshold"
    result = BytesIO()
    doc = SimpleDocTemplate(
        result, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=60, bottomMargin=57,
        title="HeatLab 2D | Numerical verification and scientific report",
        author="HeatLab 2D", subject="Analytical references, refinement and energy conservation",
    )
    story = []

    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))

    def page(number, title, subtitle):
        if number > 1:
            story.append(PageBreak())
        p(f"VERIFICATION STUDY  /  {number:02d}", "kicker")
        p(title, "title")
        p(subtitle, "sub")

    def table(header, rows, widths):
        content = [[Paragraph(str(value), styles["th"]) for value in header]]
        content.extend([Paragraph(str(value), styles["cell"]) for value in row] for row in rows)
        data = Table(content, colWidths=[CONTENT * width for width in widths],
                     hAlign="LEFT", repeatRows=1)
        data.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ]))
        story.extend((data, Spacer(1, 10)))

    def image(filename, caption, width=CONTENT - 8):
        plot = Image(BytesIO(figures[filename]), width=width, height=width * 3.18 / 8.6)
        plot.hAlign = "CENTER"
        story.extend((plot, Spacer(1, 5)))
        p(caption, "small")

    def persian_summary():
        import arabic_reshaper
        from bidi.algorithm import get_display

        lines = (
            "چکیدهٔ فارسی",
            "حل رسانش گرما با جواب‌های تحلیلی مقایسه و موازنهٔ انرژی بررسی شد.",
            "خطای شبکه و گام زمانی جداگانه سنجیده شد. نتیجه‌ها به مسائل مرجع این گزارش محدودند.",
        )
        cells = [[Paragraph(get_display(arabic_reshaper.reshape(line)), styles["persian"])]
                 for line in lines]
        block = Table(cells, colWidths=[CONTENT], hAlign="LEFT")
        block.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE),
                                   ("TOPPADDING", (0, 0), (-1, 0), 8),
                                   ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
        story.extend((block, Spacer(1, 9)))

    page(1, "Numerical Verification<br/>&amp; Scientific Report",
         "Steady and transient conduction in a rectangular solid. Analytical comparisons, "
         "mesh and time-step sensitivity, and energy conservation.")
    p("Objective", "heading")
    p("Evaluate the production finite-volume solver against analytical temperature fields and "
      "independently audit heat input and stored energy. Three prescribed benchmark problems "
      "cover internal generation, two-dimensional transient diffusion, and mixed boundary loads.")
    table(("Finest spatial comparison", "Finest temporal comparison", "Energy audit"),
          [(f"Maximum error<br/><b>{spatial['max_temperature_error_k']:.4g} K</b>",
            f"Maximum error vs exact-in-time FV<br/><b>{temporal['max_temperature_error_k']:.4g} K</b>",
            f"Max. integrated residual<br/><b>{audit_c['max_abs_energy_residual_j']:.3g} J</b>")],
          (0.32, 0.37, 0.31))
    persian_summary()
    p("Physical model", "heading")
    p("The material is homogeneous and isotropic with constant thermal properties. "
      "Temperature is uniform through the specified thickness. Both thickness faces are adiabatic. "
      "The in-plane conditions and volumetric source are time-independent. Radiation and contact "
      "resistance are outside this model.")
    p("q″ = −k ∇T &nbsp;&nbsp; | &nbsp;&nbsp; ρ c<sub>p</sub> ∂T/∂t = ∇·(k∇T) + q̇", "equation")
    p("The steady limit sets ∂T/∂t = 0. Positive boundary heat flux points into the solid. "
      "Fourier conduction, the diffusion equation and boundary/initial conditions follow [1, §§2.1, 2.3-2.4].")
    p("Evidence", "heading")
    p("Numerical fields, CSV tables and figures are generated by the same validation run. "
      "Input parameters are prescribed analytical benchmark values. Comparisons are numerical "
      "verification for the stated model and cases.")

    page(2, "Methods and run ledger", "Inputs and governing relations required to reproduce the study.")
    p("Cell-centred finite volumes", "heading")
    p("Interior faces use kA/Δ. A fixed-temperature face uses 2kA/Δ. For convection, "
      "the half-cell and film resistances give A/[Δ/(2k) + 1/h]. Flux faces add q″<sub>in</sub>A. "
      "Each volume adds q̇V. The sparse system is factorized with SciPy SuperLU [2]. "
      "Case B and case C reuse one Backward Euler factorization per run.")
    p("Steady: Kθ = b &nbsp;&nbsp; | &nbsp;&nbsp; "
      "(K + C/Δt I) Δθ = b − Kθ<sub>old</sub>, &nbsp; C = ρc<sub>p</sub>V", "equation")
    grid = run.steady.grid
    mat = run.steady.material
    p(f"All cases use L<sub>x</sub> = {grid.width_m:g} m, L<sub>y</sub> = {grid.height_m:g} m and "
      f"thickness {grid.thickness_m:g} m. Material constants are k = {mat.conductivity_w_mk:g} W/(m K), "
      f"ρ = {mat.density_kg_m3:g} kg/m³ and c<sub>p</sub> = {mat.specific_heat_j_kgk:g} J/(kg K).")
    rows = [(side.name.title(), *(_boundary(case, side) for case in (run.steady, run.transient, run.mixed)))
            for side in Side]
    rows.extend([
        ("Source [W/m³]", *(f"{case.generation_w_m3[0,0]:g}" for case in (run.steady, run.transient, run.mixed))),
        ("Initial temperature", "Steady solution", "300 + 30 cos(πx/L<sub>x</sub>) cos(πy/L<sub>y</sub>) K",
         f"Uniform {run.mixed.temperature_k[0,0,0]:g} K"),
        ("Grid", "10×5 to 80×40", f"{run.transient.grid.nx}×{run.transient.grid.ny}",
         f"{run.mixed.grid.nx}×{run.mixed.grid.ny}"),
        ("Time settings", "Steady", f"t = {run.transient.times_s[-1]:g} s; "
         f"{run.temporal[0]['steps']} to {temporal['steps']} steps",
         f"t = {run.mixed.times_s[-1]:g} s; Δt = {run.mixed.times_s[1]:g} s"),
    ])
    table(("Input", "A / Generated slab", "B / Insulated mode", "C / Mixed energy"), rows,
          (0.17, 0.255, 0.30, 0.275))
    p("Error and refinement metrics", "heading")
    p("At matched cell centres, e<sub>i</sub> = T<sub>i,num</sub> − T<sub>i,ref</sub>. "
      "Uniform cell volumes give equal weights to the N samples.")
    p("MAE = Σ|e<sub>i</sub>|/N &nbsp; | &nbsp; RMSE = √(Σe<sub>i</sub>²/N) "
      "&nbsp; | &nbsp; MaxAE = max |e<sub>i</sub>|", "equation")
    p("Observed order = log₂(e<sub>coarse,max</sub>/e<sub>fine,max</sub>) for a refinement ratio of two. "
      "Paired changes are measured separately from errors to the reference.")

    page(3, "Spatial validation", "Case A  /  Steady plane wall with uniform volumetric heat generation.")
    p("The analytical temperature follows from k d²T/dx² + q̇ = 0 with equal prescribed temperatures "
      "at x = 0 and x = L<sub>x</sub>. The y-normal boundaries are adiabatic [1, §3.5.1].")
    p("T<sub>exact</sub>(x) = T<sub>s</sub> + q̇ x(L<sub>x</sub> − x)/(2k)", "equation")
    image("steady_error.png", "Figure 1. The numerical samples coincide visually with the analytical profile. "
          "The signed-error map resolves the small positive offset on the finest mesh.")
    table(("Grid", "h [m]", "MAE [K]", "RMSE [K]", "MaxAE [K]", "Order", "Paired Δ [K]"),
          [(f"{r['nx']}×{r['ny']}", f"{r['dx_m']:.4g}", f"{r['mae_k']:.4g}", f"{r['rmse_k']:.4g}",
            f"{r['max_temperature_error_k']:.4g}", "n/a" if r["order"] is None else f"{r['order']:.3f}",
            "n/a" if r["pair_max_delta_k"] is None else f"{r['pair_max_delta_k']:.4g}")
           for r in run.spatial], (0.13, 0.12, 0.15, 0.15, 0.15, 0.11, 0.19))
    p("Grid independence criterion", "heading")
    p("For each refinement, bilinear interpolation of the four neighboring fine-cell values "
      "gives the value at each coarse-cell centre. The maximum absolute change on those common "
      f"locations for the final pair was <b>{spatial['pair_max_delta_k']:.5g} K</b>. "
      f"{verdict(spatial['pair_pass'])} of {acceptance['grid_pair_max_delta_k']:.3g} K.")
    p(f"The last measured spatial order was {spatial['order']:.3f} for this benchmark.")

    page(4, "Temporal validation", "Case B  /  Decay of a two-dimensional cosine mode with insulated edges.")
    p("The analytical cosine product satisfies zero normal gradient on each side. The separable "
      "diffusion mode follows the conduction equation [1, §§2.3, 4.2 and Ch. 5]. "
      "The exact-in-time FV reference changes only the spatial eigenvalue to isolate time-integration error.")
    p("T = 300 + 30 cos(πx/L<sub>x</sub>) cos(πy/L<sub>y</sub>) exp(−λt), "
      "&nbsp; λ = α π²(L<sub>x</sub><super>−2</super> + L<sub>y</sub><super>−2</super>)", "equation")
    p("λ<sub>h</sub> = 4α[sin²(π/(2N<sub>x</sub>))/Δx² + sin²(π/(2N<sub>y</sub>))/Δy²], "
      "&nbsp; α = k/(ρ c<sub>p</sub>)", "equation")
    image("transient_error.png", "Figure 2. Probe histories against both references and the final signed error "
          "relative to the continuum solution. The comparison uses cell-centre samples.")
    table(("Steps", "Δt [s]", "MAE [K]", "RMSE [K]", "MaxAE [K]", "Order", "Paired Δ [K]"),
          [(r["steps"], f"{r['dt_s']:.5g}", f"{r['mae_k']:.4g}", f"{r['rmse_k']:.4g}",
            f"{r['max_temperature_error_k']:.4g}", "n/a" if r["order"] is None else f"{r['order']:.3f}",
            "n/a" if r["pair_max_delta_k"] is None else f"{r['pair_max_delta_k']:.4g}")
           for r in run.temporal], (0.10, 0.13, 0.15, 0.15, 0.15, 0.105, 0.215))
    p("Table errors use the exact-in-time FV reference. On the same final field, the continuum "
      f"comparison gives MAE {temporal['continuum_mae_k']:.4g} K, RMSE {temporal['continuum_rmse_k']:.4g} K "
      f"and MaxAE {temporal['continuum_max_temperature_error_k']:.4g} K.", "small")

    page(5, "Convergence and energy", "Separation of discretization effects and independent conservation checks.")
    image("convergence.png", "Figure 3. Spatial and temporal error under refinement. The continuum error "
          "approaches a spatial floor when only Δt is reduced.", CONTENT - 52)
    p(f"At the same final time and on the same grid, the last two time-step levels changed by "
      f"<b>{temporal['pair_max_delta_k']:.5g} K</b>. {verdict(temporal['pair_pass'])} of "
      f"{acceptance['time_pair_max_delta_k']:.3g} K. The observed temporal order was {temporal['order']:.3f}.")
    p("Energy audit", "heading")
    p("r<sub>Q</sub> = ΣQ̇<sub>in</sub> + Q̇<sub>gen</sub> − (E<sub>n</sub> − E<sub>n−1</sub>)/Δt; "
      "&nbsp; r<sub>E</sub> = Σ Δt(ΣQ̇<sub>in</sub> + Q̇<sub>gen</sub>) − ΔE", "equation")
    p("The steady storage term is zero. The transient audit uses exported side powers and stored "
      "energy changes with right-endpoint integration, consistent with Backward Euler [1, §1.3.1].", "small")
    image("energy_balance.png", "Figure 4. Case C integrated energy terms and the recomputed stepwise power "
          "residual. The residual axis is in picowatts.", CONTENT - 52)
    table(("Conservation check", "Measured residual", "Absolute threshold"),
          [("Steady signed power", f"{audit_a['signed_power_residual_w']:.3g} W",
            f"{acceptance['absolute_power_residual_w']:.1g} W"),
           ("Transient maximum |power|", f"{audit_c['max_abs_power_residual_w']:.3g} W",
            f"{acceptance['absolute_power_residual_w']:.1g} W"),
           ("Transient maximum |energy|", f"{audit_c['max_abs_energy_residual_j']:.3g} J",
            f"{acceptance['absolute_energy_residual_j']:.1g} J")], (0.46, 0.28, 0.26))

    page(6, "Limitations and conclusions", "Scope of inference, reproducibility, and bibliography.")
    p("Interpretation of the energy terms", "heading")
    p(f"In case A the solid generated {audit_a['generated_w']:.4g} W, with "
      f"{-sum(audit_a['signed_boundary_inward_w']):.4g} W net outward. "
      f"In case C the net integrated boundary input was {audit_c['boundary_energy_j']:.5g} J, "
      f"the generated energy was {audit_c['generated_energy_j']:.5g} J and the stored energy increased "
      f"by {audit_c['stored_energy_change_j']:.5g} J.")
    p("Limitations", "heading")
    p("The finest-pair tests use a project demonstration tolerance for these cases. Geometry, materials, "
      "boundary complexity and accuracy requirements can change an appropriate tolerance. The plane-wall "
      "profile is one-dimensional within the two-dimensional mesh. The insulated cosine mode exercises both "
      "spatial directions. The mixed-boundary case tests conservation under combined loads.")
    p("The continuum error of the transient case retains spatial discretization error on its fixed grid. "
      "Energy conservation and analytical temperature accuracy measure different properties. "
      "The scope is computational verification for homogeneous, constant-property solids.")
    p("Conclusions", "heading")
    p(f"The final observed orders were <b>{spatial['order']:.3f}</b> in space and "
      f"<b>{temporal['order']:.3f}</b> in time. Final paired changes were "
      f"{spatial['pair_max_delta_k']:.5g} K and {temporal['pair_max_delta_k']:.5g} K. "
      f"Steady and transient energy checks were {verdict(audit_a['pass']).lower()} and "
      f"{verdict(audit_c['pass']).lower()}, respectively, for the specified tolerances.")
    p("Reproducibility", "heading")
    p("Run <b>python -m heatlab.scientific_reporting --output validation-report</b> from the project "
      "directory. The accompanying manifest stores the model assumptions, boundary and initial data, "
      "material constants, solver settings, error metrics and measured balances.")
    p("References", "heading")
    p("[1] T. L. Bergman, A. S. Lavine, F. P. Incropera, and D. P. DeWitt. "
      "<i>Fundamentals of Heat and Mass Transfer</i>, 8th ed., Wiley. "
      "§1.3.1 (energy conservation), §§2.1, 2.3-2.4 (conduction and conditions), §3.5.1 "
      "(plane wall with generation), §§4.2, 4.4.3 and 4.5.2 (separation, discrete energy balances "
      f"and numerical verification), and Ch. 5 (transient conduction). "
      f'<link href="{WILEY_TOC}" color="#087F8C">Publisher table of contents</link>.', "small")
    p("[2] SciPy documentation. <i>scipy.sparse.linalg.splu</i>. "
      '<link href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html" '
      'color="#087F8C">Sparse LU factorization</link>.', "small")

    if current is not None:
        data = current_run_manifest(current)
        page(7, "Current simulation appendix", "Recorded model and energy audit for the selected simulation.")
        grid, material = current.grid, current.material
        generated = current.generation_w_m3
        rows = [
            ("Regime", data["mode"].title()),
            ("Domain and thickness", f"{grid.width_m:g} × {grid.height_m:g} × {grid.thickness_m:g} m"),
            ("Grid", f"{grid.nx}×{grid.ny}; Δx = {grid.dx:.5g} m; Δy = {grid.dy:.5g} m"),
            ("Material constants", f"k = {material.conductivity_w_mk:g} W/(m K); "
             f"ρ = {material.density_kg_m3:g} kg/m³; c<sub>p</sub> = "
             f"{material.specific_heat_j_kgk:g} J/(kg K)"),
            ("Source range", f"{generated.min():.5g} to {generated.max():.5g} W/m³"),
            *((side.name.title(), _boundary(current, side)) for side in Side),
        ]
        if current.times_s is not None:
            initial = current.temperature_k[0]
            rows.extend([
                ("Initial temperature range", f"{initial.min():.6g} to {initial.max():.6g} K"),
                ("Time integration", f"Backward Euler; Δt = {current.times_s[1]:.6g} s; "
                 f"{len(current.times_s) - 1} steps; final time {current.times_s[-1]:.6g} s"),
            ])
        elif current.reference_mean_k is not None:
            rows.append(("Reference mean", f"{current.reference_mean_k:.6g} K"))
        table(("Input", "Recorded value"), rows, (0.32, 0.68))
        p("Final field and conservation", "heading")
        final = data["snapshot"]
        audit = data["independent_energy_audit"]
        p(f"Final temperature ranged from {final['min_k']:.6g} to {final['max_k']:.6g} K. "
          f"Maximum absolute independently recomputed power residual was "
          f"{audit['max_abs_power_residual_w']:.4g} W.")
        if audit["max_abs_integrated_residual_j"] is not None:
            p(f"Maximum absolute cumulative energy residual was "
              f"{audit['max_abs_integrated_residual_j']:.4g} J.")
        p("The analytical comparisons in the preceding sections use the reference benchmark models. "
          "The appendix documents the selected simulation and its conservation audit. Full input arrays "
          "and final summaries are recorded in current_run.json.", "small")

    doc.build(story, onFirstPage=_chrome, onLaterPages=_chrome)
    return result.getvalue()
