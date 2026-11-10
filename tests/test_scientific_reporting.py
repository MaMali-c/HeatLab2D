"""Verification evidence and exports generated from the production solver."""

import csv
from io import BytesIO, StringIO
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pypdf import PdfReader
import pytest

from heatlab.scientific_reporting import REPORT_NAME, build_report_files, report_zip
from heatlab.scientific_validation import audit_energy, error_metrics, run_validation
from heatlab.boundaries import Boundaries, Boundary, Kind
from heatlab.geometry import Grid
from heatlab.materials import Material
from heatlab.solver import solve_transient


@pytest.fixture(scope="module")
def run():
    return run_validation()


def test_metrics_match_definitions_and_require_colocated_shapes():
    numerical = np.array([[300, 301], [302, 303]])
    reference = np.full((2, 2), 300)
    metrics = error_metrics(numerical, reference)
    assert metrics["mae_k"] == pytest.approx(1.5)
    assert metrics["rmse_k"] == pytest.approx(np.sqrt(3.5))
    assert metrics["max_temperature_error_k"] == 3
    assert error_metrics(reference, reference)["rmse_k"] == 0
    with pytest.raises(ValueError, match="same points"):
        error_metrics(reference, np.full(2, 300))


def test_analytical_and_independence_checks_are_distinct(run):
    assert len(run.spatial) == 4
    assert len(run.temporal) == 7
    assert run.spatial[-1]["order"] == pytest.approx(2.0, abs=1e-6)
    assert run.temporal[-1]["order"] == pytest.approx(1.0, abs=0.005)
    assert run.spatial[-1]["pair_pass"] and run.temporal[-1]["pair_pass"]
    assert not run.spatial[-2]["pair_pass"] and not run.temporal[-2]["pair_pass"]
    assert run.temporal[-1]["continuum_max_temperature_error_k"] > 0.08
    assert run.temporal[-1]["max_temperature_error_k"] < 0.01


def test_energy_audit_reconstructs_source_and_storage(run):
    steady = audit_energy(run.steady)
    transient = audit_energy(run.mixed)
    assert steady.passed and transient.passed
    assert steady.power_residual_w[0] == pytest.approx(
        run.steady.inward_w.sum() + run.steady.generated_w, abs=1e-16
    )
    assert np.isnan(transient.power_residual_w[0])
    np.testing.assert_array_equal(transient.storage_w[1:],
                                  np.diff(run.mixed.energy_change_j) / np.diff(run.mixed.times_s))
    assert transient.max_abs_energy_j < 1e-8


def test_energy_audit_keeps_changes_smaller_than_absolute_energy_resolution():
    insulated = Boundary(Kind.INSULATED)
    result = solve_transient(Grid(1, 1, 1, 4, 4), Material(100, 8000, 500),
                             Boundaries(insulated, insulated, insulated, insulated),
                             initial_k=300, duration_s=1e-10, steps=4, generation_w_m3=30)
    assert result.stored_energy_j[-1] == result.stored_energy_j[0]
    np.testing.assert_allclose(result.energy_change_j, result.times_s * 30, rtol=1e-12, atol=0)
    assert audit_energy(result).max_abs_power_w < 1e-10


def test_report_bundle_contains_matched_fields_and_figures(run):
    files = build_report_files(run)
    pdf = PdfReader(BytesIO(files[REPORT_NAME]))
    assert len(pdf.pages) == 6
    assert "Objective" in pdf.pages[0].extract_text()
    assert "Incropera" in pdf.pages[-1].extract_text()
    assert "Limitations" in pdf.pages[-1].extract_text()
    assert pdf.pages[-1].get("/Annots")
    assert all(len(page.extract_text().strip()) > 350 for page in pdf.pages)

    names = [name for name in files if name.endswith(".png")]
    assert len(names) == 4
    for name in names:
        with Image.open(BytesIO(files[name])) as figure:
            assert figure.format == "PNG" and figure.width > 1500

    manifest = json.loads(files["validation.json"])
    assert manifest["models"]["A_steady_generated_slab"]["grid"]["nx"] == run.steady.grid.nx
    assert manifest["time_convergence"][-1]["rmse_k"] == run.temporal[-1]["rmse_k"]
    rows = list(csv.DictReader(StringIO(files["data/transient_field.csv"].decode())))
    assert len(rows) == run.transient.grid.size
    assert float(rows[0]["numerical_k"]) == run.transient.temperature_k[-1, 0, 0]
    assert float(rows[0]["continuum_k"]) == run.transient_continuum[0, 0]

    with ZipFile(BytesIO(report_zip(files))) as bundle:
        assert set(bundle.namelist()) == set(files)
        assert bundle.read(REPORT_NAME) == files[REPORT_NAME]


def test_optional_current_simulation_appendix(run):
    files = build_report_files(run, current=run.mixed)
    pdf = PdfReader(BytesIO(files[REPORT_NAME]))
    assert len(pdf.pages) == 7
    assert "Current simulation appendix" in pdf.pages[-1].extract_text()
    current = json.loads(files["current_run.json"])
    assert current["time"]["initial_temperature_k"] == 300
    assert current["boundaries"]["right"]["h_w_m2k"] == 25
    assert "current_simulation" in files["data/energy_audit.csv"].decode()
