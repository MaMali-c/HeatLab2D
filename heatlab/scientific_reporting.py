"""Export a reproducible validation PDF, numerical CSV tables, and PNG figures."""

import argparse
import csv
from importlib.metadata import version
from io import BytesIO, StringIO
import json
from pathlib import Path
from platform import python_version
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .report_figures import report_figures
from .scientific_validation import EnergyAudit, ValidationRun, audit_energy, current_run_manifest
from .scientific_validation import run_validation, validation_manifest
from .solver import Result
from .visualization import energy_history

REPORT_NAME = "HeatLab2D-Scientific-Validation.pdf"


def _csv(header, rows) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _table_csv(rows: list[dict]) -> bytes:
    columns = list(rows[0])
    return _csv(columns, ([row[column] for column in columns] for row in rows))


def _field_csv(result: Result, reference: np.ndarray,
               semidiscrete: np.ndarray | None = None) -> bytes:
    header = ["x_m", "y_m", "numerical_k", "continuum_k", "signed_error_vs_continuum_k"]
    grid = result.grid
    numerical = result.temperature_k[-1]
    if semidiscrete is not None:
        header.extend(["exact_in_time_fv_k", "signed_error_vs_exact_in_time_fv_k"])
    x, y = grid.x_m, grid.y_m

    def rows():
        for j in range(grid.ny):
            for i in range(grid.nx):
                t, exact = numerical[j, i], reference[j, i]
                row = [x[i], y[j], t, exact, t - exact]
                if semidiscrete is not None:
                    row.extend((semidiscrete[j, i], t - semidiscrete[j, i]))
                yield row

    return _csv(header, rows())


def _energy_rows(case: str, result: Result, audit: EnergyAudit):
    energy = None if result.times_s is None else energy_history(result)
    for index, side in enumerate(result.inward_w):
        has_rate = energy is None or index > 0
        yield (
            case, None if result.times_s is None else result.times_s[index],
            *side, result.generated_w, audit.storage_w[index] if has_rate else None,
            audit.power_residual_w[index] if has_rate else None,
            None if energy is None else energy.boundary_j[index],
            None if energy is None else energy.generated_j[index],
            None if energy is None else energy.change_j[index],
            None if energy is None else energy.residual_j[index],
        )


def _energy_csv(run: ValidationRun, current: Result | None) -> bytes:
    cases = [("A_steady_generated_slab", run.steady, run.steady_energy),
             ("C_transient_mixed_energy", run.mixed, run.transient_energy)]
    if current is not None:
        cases.append(("current_simulation", current, audit_energy(current)))
    return _csv(
        ("case", "time_s", "left_inward_w", "right_inward_w", "bottom_inward_w", "top_inward_w",
         "generated_w", "storage_w", "signed_power_residual_w", "boundary_energy_j",
         "generated_energy_j", "stored_energy_change_j", "signed_integrated_residual_j"),
        (row for case, result, audit in cases for row in _energy_rows(case, result, audit)),
    )


def _json(data: dict) -> bytes:
    return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8")


def build_report_files(run: ValidationRun, current: Result | None = None) -> dict[str, bytes]:
    """Produce the report and its traceable inputs and outputs from one study."""
    from .report_pdf import build_pdf

    figures = report_figures(run)
    manifest = validation_manifest(run)
    manifest["software"] = {
        "python": python_version(),
        **{name: version(name) for name in ("numpy", "scipy", "matplotlib", "reportlab")},
    }
    files = {
        REPORT_NAME: build_pdf(run, figures, current),
        "validation.json": _json(manifest),
        "data/steady_grid.csv": _table_csv(run.spatial),
        "data/transient_time.csv": _table_csv(run.temporal),
        "data/steady_field.csv": _field_csv(run.steady, run.steady_exact),
        "data/transient_field.csv": _field_csv(run.transient, run.transient_continuum,
                                               run.transient_semidiscrete),
        "data/energy_audit.csv": _energy_csv(run, current),
        **{f"figures/{filename}": data for filename, data in figures.items()},
    }
    if current is not None:
        files["current_run.json"] = _json(current_run_manifest(current))
    return files


def report_zip(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation-report"),
                        help="Directory for the PDF, CSV, PNG and JSON files.")
    output = parser.parse_args().output
    files = build_report_files(run_validation())
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "HeatLab2D-Validation-Data.zip").write_bytes(report_zip(files))
    print(f"Wrote {len(files)} files and a ZIP to {output}")


if __name__ == "__main__":
    main()
