"""Stress tests, the dashboard overview endpoint and the static dashboard."""
import unittest

from app.bootstrap import build_system
from app.service import RiskService
from app.stress import STANDARD_SCENARIOS, validate


class StressMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = build_system(":memory:", model_path=None, baseline_ticks=100, seed=303)
        cls.svc = RiskService(cls.s)

    def test_market_shock_is_beta_times_move(self):
        book = self.s.tools.get_positions("BAL")
        beta = {r["symbol"]: r["beta"] for r in self.s.db.query("SELECT symbol, beta FROM instruments")}
        expected = sum(p["market_value"] * beta[p["symbol"]] * -0.10 for p in book["positions"])
        out = self.svc.stress("BAL", [{"name": "m", "market": -0.10, "sectors": {}}])
        self.assertAlmostEqual(out["scenarios"][0]["pnl"], expected, delta=1.0)

    def test_market_shock_is_symmetric_and_linear(self):
        out = self.svc.stress("TECH", [{"market": -0.10}, {"market": 0.10}, {"market": -0.20}])
        down, up, crash = (s["pnl"] for s in out["scenarios"])
        self.assertAlmostEqual(down, -up, delta=1.0)
        self.assertAlmostEqual(crash, 2 * down, delta=1.0)

    def test_sector_shock_only_hits_that_sector(self):
        out = self.svc.stress("TECH", [{"sectors": {"Energy": -0.20}}])
        self.assertTrue(all(c["sector"] == "Energy" or c["pnl"] == 0 for c in out["scenarios"][0]["worst_contributors"]))

    def test_standard_scenarios_and_worst_contributors(self):
        out = self.svc.stress("NEUT")
        self.assertEqual(len(out["scenarios"]), len(STANDARD_SCENARIOS))
        for s in out["scenarios"]:
            self.assertLessEqual(len(s["worst_contributors"]), 3)
            pnls = [c["pnl"] for c in s["worst_contributors"]]
            self.assertEqual(pnls, sorted(pnls))

    def test_validation(self):
        with self.assertRaises(ValueError):
            validate({"sectors": {"Crypto": -0.1}})
        with self.assertRaises(ValueError):
            validate({"market": -1.5})
        self.assertEqual(validate({})["name"], "Custom scenario")

    def test_overview_shape(self):
        ov = self.svc.overview(n=50)
        self.assertEqual({b["portfolio_id"] for b in ov["portfolios"]}, {"BAL", "MACRO", "NEUT", "TECH"})
        hist = ov["portfolios"][0]["history"]
        self.assertEqual(len(hist), 50)
        self.assertEqual([h["tick"] for h in hist], sorted(h["tick"] for h in hist))


class DashboardApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import create_app
        cls.client = TestClient(create_app(build_system(":memory:", model_path=None, baseline_ticks=100, seed=404)))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_standard_stress(self):
        r = self.client.get("/portfolios/TECH/stress")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["scenarios"]), len(STANDARD_SCENARIOS))

    def test_custom_stress(self):
        r = self.client.post("/portfolios/BAL/stress",
                             json={"scenarios": [{"name": "mine", "market": -0.05, "sectors": {"Tech": -0.1}}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["scenarios"][0]["name"], "mine")

    def test_bad_stress_requests(self):
        self.assertEqual(self.client.post("/portfolios/BAL/stress",
                                          json={"scenarios": [{"sectors": {"Crypto": -0.1}}]}).status_code, 422)
        self.assertEqual(self.client.post("/portfolios/BAL/stress",
                                          json={"scenarios": [{"market": -2}]}).status_code, 422)
        self.assertEqual(self.client.get("/portfolios/NOPE/stress").status_code, 404)

    def test_overview_and_dashboard_page(self):
        r = self.client.get("/dashboard/overview?n=30")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["portfolios"][0]["history"]), 30)
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Risk Copilot", page.text)
        self.assertEqual(self.client.get("/app.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
