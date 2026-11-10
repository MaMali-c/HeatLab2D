"""Check that exported data comes from the selected result."""

import csv
import json
import unittest

import numpy as np

from heatlab.boundaries import Boundaries, Boundary, Kind
from heatlab.geometry import Grid
from heatlab.materials import Material
from heatlab.report import field_csv, history_csv, report_json
from heatlab.solver import solve_steady, solve_transient


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(0.2, 0.1, 0.01, 4, 2)
        self.material = Material(15, 7800, 500)
        insulated = Boundary(Kind.INSULATED)
        self.bc = Boundaries(Boundary(Kind.TEMPERATURE, 350), Boundary(Kind.CONVECTION, 300, 10),
                             insulated, insulated)

    def test_field_and_steady_metadata(self):
        result = solve_steady(self.grid, self.material, self.bc)
        rows = list(csv.DictReader(field_csv(result, 0).splitlines()))
        self.assertEqual(len(rows), 8)
        np.testing.assert_array_equal([float(row["temperature_k"]) for row in rows],
                                      result.temperature_k[0].ravel())
        self.assertAlmostEqual(float(rows[-1]["x_m"]), self.grid.x_m[-1])
        self.assertAlmostEqual(float(rows[-1]["y_m"]), self.grid.y_m[-1])
        report = json.loads(report_json(result, 0))
        self.assertEqual(report["mode"], "steady")
        self.assertEqual(report["boundaries"]["right"]["h_w_m2k"], 10)
        self.assertEqual(report["snapshot"]["mean_k"], result.temperature_k[0].mean())

    def test_transient_initial_and_final_export(self):
        initial = 300 + np.arange(8).reshape(self.grid.shape)
        result = solve_transient(self.grid, self.material, self.bc, initial_k=initial,
                                 duration_s=16, steps=4)
        rows = list(csv.DictReader(history_csv(result).splitlines()))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["balance_w"], "")
        self.assertEqual(float(rows[-1]["time_s"]), 16)
        self.assertAlmostEqual(float(rows[-1]["storage_w"]), result.storage_w[-1])
        first = json.loads(report_json(result, 0))
        last = json.loads(report_json(result, 4))
        self.assertIsNone(first["snapshot"]["balance_w"])
        self.assertEqual(last["time"]["initial_temperature_k"], initial.tolist())
        self.assertEqual(last["snapshot"]["time_s"], 16)
        self.assertEqual(last["snapshot"]["max_k"], result.temperature_k[-1].max())
