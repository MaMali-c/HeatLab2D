"""Exercise the bilingual dashboard with real solves."""

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from heatlab.boundaries import Kind

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=45).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_startup_solves_transient_preset_and_switches_to_steady(self):
        preset = self.app.session_state["simulation"][1]
        self.assertEqual(preset.times_s[self.app.session_state["frame"]], 120)
        self.assertEqual(len(self.app.get("plotly_chart")), 3)

        self.app.button_group(key="mode").set_value("Steady (پایا)").run()
        self.assertIn("Inputs changed", self.app.info[0].value)
        self.assertEqual(len(self.app.metric), 0)
        self.app.selectbox(key="RIGHT_kind").set_value(Kind.TEMPERATURE).run()
        self.app.number_input(key="LEFT_surface").set_value(350.0)
        self.app.number_input(key="RIGHT_surface").set_value(300.0)
        self.app.button(key="run").click().run()
        self.assertEqual(len(self.app.exception), 0)
        result = self.app.session_state["simulation"][1]
        self.assertIsNone(result.times_s)
        self.assertAlmostEqual(result.temperature_k.mean(), 325.0)

    def test_time_steps_probes_and_temperature_units(self):
        self.app.selectbox(key="speed_choice").set_value(4.0).run()
        self.assertEqual(self.app.session_state["playback_speed"], 4.0)
        self.app.button(key="next_frame").click().run()
        self.assertEqual(self.app.session_state["frame"], 25)
        self.app.button(key="prev_frame").click().run()
        self.assertEqual(self.app.session_state["frame"], 24)
        self.app.slider(key="snapshot").set_value(0).run()
        self.assertEqual(self.app.session_state["frame"], 0)
        self.assertTrue(all(m.value == "25.00 °C" for m in self.app.metric if "شار" not in m.label))
        self.app.selectbox(key="display_unit").set_value("K").run()
        self.assertTrue(any(m.value == "298.15 K" for m in self.app.metric))
        self.app.number_input(key="_probe_x").set_value(0.15).run()
        self.assertEqual(self.app.session_state["probes"][0][1], 36)
        self.assertEqual(len(self.app.exception), 0)

    def test_layer_controls_change_real_plot_traces(self):
        chart = self.app.get("plotly_chart")[0]
        types = [trace["type"] for trace in json.loads(chart.proto.spec)["data"]]
        self.assertIn("heatmap", types)
        self.assertIn("contour", types)
        self.app.toggle(key="layer_heatmap").set_value(False).run()
        types = [trace["type"] for trace in json.loads(self.app.get("plotly_chart")[0].proto.spec)["data"]]
        self.assertNotIn("heatmap", types)
        self.app.toggle(key="layer_surface").set_value(False).run()
        self.assertEqual(len(self.app.get("plotly_chart")), 2)

    def test_analysis_exports_and_validation_benchmarks(self):
        self.app.button(key="nav_Analysis").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.get("plotly_chart")), 2)
        self.app.button_group(key="analysis_view").set_value("Export (خروجی)").run()
        self.assertEqual(len(self.app.get("download_button")), 6)
        self.app.button(key="prepare_png").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(self.app.session_state["media"][1].startswith(b"\x89PNG"))
        self.app.slider(key="analysis_frame").set_value(1).run()
        self.assertEqual(len(self.app.get("download_button")), 6)
        self.app.button(key="nav_Validation").click().run()
        self.app.button(key="benchmark").click().run()
        self.assertEqual(len(self.app.exception), 0)
        run = self.app.session_state["validation_run"]
        self.assertAlmostEqual(run.spatial[-1]["order"], 2.0, places=6)
        self.assertGreater(run.temporal[-1]["order"], 0.98)
        self.assertLess(run.transient_energy.max_abs_energy_j, 1e-8)
        self.app.button(key="validation_prepare").click().run()
        self.assertEqual(len(self.app.exception), 0)
        _, pdf, bundle = self.app.session_state["validation_report"]
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertTrue(bundle.startswith(b"PK"))
        self.assertEqual(len(self.app.get("download_button")), 2)

    def test_companion_solve_and_comparison_frame(self):
        self.app.button(key="nav_Analysis").click().run()
        self.app.button_group(key="analysis_view").set_value("Compare (مقایسه)").run()
        self.app.button(key="compare_solve").click().run()
        self.assertEqual(len(self.app.exception), 0)
        comparison = self.app.session_state["comparison"][1]
        self.assertIs(comparison.transient, self.app.session_state["simulation"][1])
        self.assertEqual(len(self.app.get("plotly_chart")), 4)
        self.app.slider(key="comparison_frame").set_value(0).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_flux_only_comparison_defaults_to_steady_reference_mean(self):
        self.app.button_group(key="mode").set_value("Steady (پایا)").run()
        self.app.selectbox(key="LEFT_kind").set_value(Kind.INSULATED).run()
        self.app.number_input(key="reference_mean").set_value(375.0)
        self.app.button(key="run").click().run()
        self.app.button(key="nav_Analysis").click().run()
        self.app.button_group(key="analysis_view").set_value("Compare (مقایسه)").run()
        self.assertEqual(self.app.number_input(key="compare_initial").value, 375.0)
        self.app.button(key="compare_solve").click().run()
        self.assertAlmostEqual(self.app.session_state["comparison"][1].rms_k[-1], 0)

    def test_transient_budget_is_visible_before_solving(self):
        self.app.number_input(key="nx").set_value(100)
        self.app.number_input(key="ny").set_value(100)
        self.app.number_input(key="steps").set_value(500).run()
        self.assertTrue(self.app.button(key="run").disabled)
        self.assertIn("499", self.app.warning[0].value)

    def test_incompatible_flux_has_readable_error(self):
        self.app.button_group(key="mode").set_value("Steady (پایا)").run()
        self.app.selectbox(key="LEFT_kind").set_value(Kind.FLUX).run()
        self.app.number_input(key="LEFT_flux").set_value(10.0)
        self.app.button(key="run").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("zero net heat input", self.app.error[0].value)
        self.assertEqual(len(self.app.metric), 0)

    def test_volumetric_generation_reaches_the_solver(self):
        self.app.number_input(key="generation").set_value(30_000.0)
        self.app.button(key="run").click().run()
        self.assertEqual(len(self.app.exception), 0)
        result = self.app.session_state["simulation"][1]
        self.assertAlmostEqual(result.generated_w, 6.0)
        self.assertLess(abs(result.balance_w[-1]), 1e-8)
