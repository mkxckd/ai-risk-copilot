"""Application service layer (framework-free so it can be unit-tested without FastAPI)."""
import json

from .anomaly import FEATURES
from .stress import run_stress


class RiskService:
    def __init__(self, system):
        self.s = system

    def _check(self, pid):
        if pid not in self.s.engine.portfolio_ids:
            raise KeyError(pid)

    def portfolios(self):
        return self.s.db.query(
            "SELECT p.portfolio_id, p.name, p.strategy, p.capital, r.tick, r.nav, r.leverage, r.var_95, "
            "r.anomaly_score, r.is_anomaly FROM portfolios p LEFT JOIN risk_snapshots r "
            "ON r.portfolio_id=p.portfolio_id AND r.tick=(SELECT MAX(tick) FROM risk_snapshots)")

    def risk(self, pid):
        self._check(pid)
        return self.s.tools.get_latest_snapshot(pid)

    def history(self, pid, n=60):
        self._check(pid)
        return self.s.tools.get_risk_history(pid, n)

    def alerts(self, pid=None, severity=None, limit=50):
        rows = self.s.db.query(
            "SELECT * FROM alerts WHERE (:p IS NULL OR portfolio_id=:p) AND (:s IS NULL OR severity=:s) "
            "ORDER BY tick DESC LIMIT :n", {"p": pid, "s": severity, "n": limit})
        return [{**r, "drivers": json.loads(r["drivers"] or "[]")} for r in rows]

    def score(self, pid, metrics: dict):
        """Stateless inference: score an arbitrary metrics vector against a portfolio's own baseline."""
        if not self.s.detector.known(pid):
            raise KeyError(pid)
        missing = [f for f in FEATURES if f not in metrics]
        if missing:
            raise ValueError(f"missing features: {missing}")
        return self.s.detector.score(pid, metrics)

    def explain(self, pid, question=None):
        self._check(pid)
        return self.s.agent.explain(pid, question)

    def stress(self, pid, scenarios=None):
        self._check(pid)
        return run_stress(self.s.tools, self.s.db, pid, scenarios)

    def overview(self, n=120):
        """Everything the dashboard needs in one call: latest risk, short history and recent alerts per portfolio."""
        out = []
        names = {r["portfolio_id"]: r for r in self.s.db.query("SELECT portfolio_id, name, strategy FROM portfolios")}
        for pid in self.s.engine.portfolio_ids:
            latest = self.s.tools.get_latest_snapshot(pid)
            hist = list(reversed(self.s.tools.get_risk_history(pid, n)["rows"]))
            out.append({"portfolio_id": pid, "name": names[pid]["name"], "strategy": names[pid]["strategy"],
                        "snapshot": latest["snapshot"], "sector_weights": latest["sector_weights"],
                        "latest_alert": latest["latest_alert"],
                        "history": [{k: r[k] for k in ("tick", "var_95_pct", "vol_ann", "leverage", "top_weight", "is_anomaly")}
                                    for r in hist]})
        return {"tick": self.s.engine.tick, "portfolios": out, "alerts": self.alerts(limit=25)}

    def analytics(self, lookback=24):
        db = self.s.db
        return {"latest": db.named("latest_snapshots"), "alerts": db.named("alert_summary"),
                "risk_change": db.named("risk_change", lookback=lookback)}

    def step(self, n=1):
        snaps = []
        for _ in range(n):
            snaps = self.s.engine.step()
        return {"tick": self.s.engine.tick, "snapshots": snaps}

    def inject(self, kind, delay=1, **params):
        self.s.engine.schedule(kind, delay=delay, **params)
        return {"scheduled": kind, "at_tick": self.s.engine.tick + max(1, delay), "params": params}
