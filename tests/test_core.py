import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.agent import RiskAgent
from app.bootstrap import build_system
from app.risk import compute_risk, component_risk
from app.service import RiskService


class RiskMathTest(unittest.TestCase):
    def test_exposures_and_var(self):
        rng = np.random.default_rng(0)
        rets = pd.DataFrame(rng.normal(0, 0.001, (200, 2)), columns=["A", "B"])
        inst = {"A": {"sector": "X", "beta": 1.0}, "B": {"sector": "Y", "beta": 0.5}}
        r = compute_risk({"A": 100.0, "B": -50.0}, {"A": 10.0, "B": 10.0}, cash=500.0, returns=rets, instruments=inst)
        self.assertAlmostEqual(r["nav"], 1000.0)          # 500 cash + 1000 long - 500 short
        self.assertAlmostEqual(r["gross"], 1500.0)
        self.assertAlmostEqual(r["leverage"], 1.5)
        self.assertAlmostEqual(r["beta"], 1.0 * 1.0 + (-0.5) * 0.5)
        self.assertGreater(r["var_99"], r["var_95"])
        self.assertAlmostEqual(component_risk(pd.Series({"A": 1.0, "B": -0.5}), rets).sum(), 1.0)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = build_system(":memory:", model_path=None, baseline_ticks=400, seed=101)
        cls.svc = RiskService(cls.s)

    def test_1_clean_data_is_quiet(self):
        e = self.s.engine
        t0 = e.tick
        e.run(200)
        flagged = self.s.db.query("SELECT COUNT(*) c FROM risk_snapshots WHERE tick>? AND is_anomaly=1", (t0,))[0]["c"]
        self.assertLessEqual(flagged / (200 * 4), 0.02)

    def test_2_concentration_is_detected_with_drivers(self):
        e = self.s.engine
        start = e.tick
        e.schedule("concentrate", portfolio_id="TECH", symbol="QBIT", target_weight=0.45)
        e.run(3)
        alerts = self.svc.alerts("TECH")
        alerts = [a for a in alerts if a["tick"] > start]
        self.assertTrue(alerts)
        self.assertIn("top_weight", [d["feature"] for d in alerts[-1]["drivers"]])

    def test_3_rule_based_explanation_is_grounded(self):
        out = self.svc.explain("TECH")
        self.assertEqual(out["mode"], "rule_based")
        text = out["explanation"]
        self.assertIn("concentration_event", text)          # cause pulled from the trade log
        self.assertIn("QBIT", text)
        self.assertTrue(any(ev["tool"] == "explain_change" for ev in out["evidence"]))

    def test_4_stateless_scoring_endpoint_logic(self):
        snap = self.svc.risk("BAL")["snapshot"]
        normal = self.svc.score("BAL", {k: snap[k] for k in
                                        ["leverage", "beta", "var_95_pct", "top_weight", "hhi", "sector_max", "vol_ann", "drawdown"]})
        self.assertFalse(normal["is_anomaly"])
        bad = self.svc.score("BAL", {"leverage": 3.0, "beta": 2.5, "var_95_pct": 0.09, "top_weight": 0.6,
                                     "hhi": 0.4, "sector_max": 0.9, "vol_ann": 0.8, "drawdown": 0.2})
        self.assertTrue(bad["is_anomaly"])
        with self.assertRaises(KeyError):
            self.svc.score("NOPE", {})

    def test_5_sql_analytics(self):
        a = self.svc.analytics()
        self.assertEqual(len(a["latest"]), 4)
        self.assertEqual(sorted(r["var_rank"] for r in a["latest"]), [1, 2, 3, 4])
        self.assertTrue(a["alerts"])
        hist = self.svc.history("MACRO", 10)["rows"]
        self.assertEqual(len(hist), 10)
        self.assertIn("var_95_ma40", hist[0])

    def test_6_llm_tool_loop_with_fake_client(self):
        calls = []

        class FakeMessages:
            def create(self, **kw):
                calls.append(kw)
                if len(calls) == 1:
                    block = SimpleNamespace(type="tool_use", id="tu_1", name="get_latest_snapshot",
                                            input={"portfolio_id": "TECH"})
                    return SimpleNamespace(stop_reason="tool_use", content=[block])
                return SimpleNamespace(stop_reason="end_turn",
                                       content=[SimpleNamespace(type="text", text="TECH risk rose due to QBIT.")])

        agent = RiskAgent(self.s.tools, client=SimpleNamespace(messages=FakeMessages()))
        out = agent.explain("TECH")
        self.assertEqual(out["mode"], "llm")
        self.assertEqual(out["evidence"][0]["tool"], "get_latest_snapshot")
        last_user = calls[1]["messages"][2]   # user prompt, assistant tool_use, user tool_result
        self.assertEqual(last_user["content"][0]["tool_use_id"], "tu_1")
        self.assertIn("snapshot", json.loads(last_user["content"][0]["content"]))

    def test_7_llm_failure_falls_back(self):
        class Boom:
            class messages:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("api down")

        out = RiskAgent(self.s.tools, client=Boom).explain("BAL")
        self.assertEqual(out["mode"], "rule_based_after_llm_error")
        self.assertIn("api down", out["llm_error"])

    def test_8_resume_from_disk(self):
        import os, tempfile
        path = os.path.join(tempfile.mkdtemp(), "r.db")
        a = build_system(path, model_path=None, baseline_ticks=200, seed=5)
        a.engine.run(5)
        tick = a.engine.tick
        b = build_system(path, model_path=None, baseline_ticks=200, seed=5)
        self.assertEqual(b.engine.tick, tick)
        b.engine.step()
        self.assertEqual(b.engine.tick, tick + 1)


if __name__ == "__main__":
    unittest.main()

class FastAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import create_app
        system = build_system(":memory:", model_path=None, baseline_ticks=100, seed=202)
        cls.client = TestClient(create_app(system))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_endpoint(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_portfolios_and_risk_endpoints(self):
        r = self.client.get("/portfolios")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 4)
        r = self.client.get("/portfolios/TECH/risk")
        self.assertEqual(r.status_code, 200)
        self.assertIn("snapshot", r.json())

    def test_unknown_portfolio_returns_404(self):
        r = self.client.get("/portfolios/NOPE/risk")
        self.assertEqual(r.status_code, 404)

    def test_history_validation(self):
        r = self.client.get("/portfolios/TECH/history?n=0")
        self.assertEqual(r.status_code, 422)

    def test_stateless_score_endpoint(self):
        risk = self.client.get("/portfolios/BAL/risk").json()["snapshot"]
        payload = {"portfolio_id": "BAL", **{k: risk[k] for k in
            ["leverage", "beta", "var_95_pct", "top_weight", "hhi", "sector_max", "vol_ann", "drawdown"]}}
        r = self.client.post("/score", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertIn("is_anomaly", r.json())

    def test_explain_endpoint(self):
        r = self.client.post("/portfolios/TECH/explain", json={"question": "Why did risk change?"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("evidence", r.json())

    def test_simulation_controls(self):
        r = self.client.post("/simulation/step?n=1")
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/simulation/inject", json={
            "kind": "concentrate", "portfolio_id": "TECH", "symbol": "QBIT",
            "target_weight": 0.45, "delay": 1
        })
        self.assertEqual(r.status_code, 200)
