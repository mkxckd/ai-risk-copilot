"""Read-only retrieval tools shared by the LLM agent and the rule-based fallback.
Every tool returns plain JSON-serialisable data so the same output can be shown as audit evidence."""
import json

import pandas as pd

METRICS = ["nav", "leverage", "beta", "var_95", "var_95_pct", "vol_ann", "drawdown",
           "top_weight", "hhi", "sector_max"]


def _r(x, n=5):
    return round(x, n) if isinstance(x, float) else x


def _round_dict(d):
    return {k: _r(v) for k, v in d.items()}


TOOL_SPECS = [
    {"name": "get_latest_snapshot",
     "description": "Latest risk snapshot for a portfolio (NAV, leverage, beta, 1-day VaR, vol, concentration, "
                    "sector exposure) plus its most recent anomaly alert and drivers.",
     "input_schema": {"type": "object", "properties": {"portfolio_id": {"type": "string"}},
                      "required": ["portfolio_id"]}},
    {"name": "get_risk_history",
     "description": "Recent risk snapshots with 40-tick moving averages (SQL window functions) to judge whether "
                    "current risk is unusual relative to its recent past.",
     "input_schema": {"type": "object", "properties": {"portfolio_id": {"type": "string"},
                                                       "n": {"type": "integer", "default": 30}},
                      "required": ["portfolio_id"]}},
    {"name": "get_positions",
     "description": "Current positions with market value, weight and sector.",
     "input_schema": {"type": "object", "properties": {"portfolio_id": {"type": "string"}},
                      "required": ["portfolio_id"]}},
    {"name": "explain_change",
     "description": "Decomposes how a portfolio's risk changed over the last `lookback` ticks: metric deltas, "
                    "trades executed, P&L attribution by symbol, weight and sector-exposure changes.",
     "input_schema": {"type": "object", "properties": {"portfolio_id": {"type": "string"},
                                                       "lookback": {"type": "integer", "default": 24}},
                      "required": ["portfolio_id"]}},
    {"name": "get_market_context",
     "description": "Market backdrop over the last `lookback` ticks: top movers, sector returns, and the ratio of "
                    "recent market volatility to its prior level.",
     "input_schema": {"type": "object", "properties": {"lookback": {"type": "integer", "default": 24}}}},
    {"name": "get_similar_alerts",
     "description": "Historical alerts, optionally filtered by portfolio and by driver feature "
                    "(leverage, beta, var_95_pct, top_weight, hhi, sector_max, vol_ann, drawdown).",
     "input_schema": {"type": "object", "properties": {"portfolio_id": {"type": "string"},
                                                       "feature": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 5}}}},
]


class RiskTools:
    def __init__(self, db):
        self.db = db

    # ---------- dispatch ----------
    def call(self, name: str, args: dict | None = None) -> dict:
        args = args or {}
        try:
            return getattr(self, name)(**args)
        except Exception as exc:  # tools must never crash the agent loop
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ---------- helpers ----------
    def _snap_at(self, pid, tick):
        rows = self.db.query("SELECT * FROM risk_snapshots WHERE portfolio_id=? AND tick<=? "
                             "ORDER BY tick DESC LIMIT 1", (pid, tick))
        if not rows:
            raise KeyError(f"no snapshot for portfolio {pid!r}")
        return rows[0]

    def _positions_at(self, pid, tick):
        rows = self.db.query("SELECT symbol, SUM(quantity) q FROM trades WHERE portfolio_id=? AND tick<=? "
                             "GROUP BY symbol HAVING ABS(SUM(quantity))>1e-9", (pid, tick))
        return {r["symbol"]: r["q"] for r in rows}

    def _prices_at(self, tick):
        rows = self.db.query("SELECT symbol, price FROM prices WHERE tick=?", (tick,))
        return {r["symbol"]: r["price"] for r in rows}

    def _sectors_at(self, pid, tick):
        rows = self.db.query("SELECT sector, net_weight FROM sector_exposures WHERE portfolio_id=? AND tick=?",
                             (pid, tick))
        return {r["sector"]: r["net_weight"] for r in rows}

    # ---------- tools ----------
    def get_latest_snapshot(self, portfolio_id):
        snap = self._snap_at(portfolio_id, 10**9)
        alerts = self.db.query("SELECT * FROM alerts WHERE portfolio_id=? ORDER BY tick DESC LIMIT 1", (portfolio_id,))
        alert = None
        if alerts:
            alert = {**alerts[0], "drivers": json.loads(alerts[0]["drivers"] or "[]")}
        return {"snapshot": _round_dict(snap), "sector_weights": _round_dict(self._sectors_at(portfolio_id, snap["tick"])),
                "latest_alert": alert}

    def get_risk_history(self, portfolio_id, n=30):
        rows = self.db.named("risk_regime", portfolio_id=portfolio_id, n=int(n))
        return {"rows": [_round_dict(r) for r in rows]}

    def get_positions(self, portfolio_id):
        snap = self._snap_at(portfolio_id, 10**9)
        pos, px = self._positions_at(portfolio_id, snap["tick"]), self._prices_at(snap["tick"])
        sector = {r["symbol"]: r["sector"] for r in self.db.query("SELECT symbol, sector FROM instruments")}
        rows = [{"symbol": s, "sector": sector[s], "quantity": _r(q, 2), "market_value": _r(q * px[s], 2),
                 "weight": _r(q * px[s] / snap["nav"])} for s, q in pos.items()]
        rows.sort(key=lambda r: -abs(r["weight"]))
        return {"tick": snap["tick"], "nav": _r(snap["nav"], 2), "positions": rows}

    def explain_change(self, portfolio_id, lookback=24):
        now = self._snap_at(portfolio_id, 10**9)
        then = self._snap_at(portfolio_id, now["tick"] - int(lookback))
        t0, t1 = then["tick"], now["tick"]
        changes = {}
        for m in METRICS:
            a, b = then[m], now[m]
            changes[m] = {"then": _r(a), "now": _r(b), "change_pct": _r(100 * (b - a) / abs(a), 2) if a else None}
        pos0, pos1 = self._positions_at(portfolio_id, t0), self._positions_at(portfolio_id, t1)
        p0, p1 = self._prices_at(t0), self._prices_at(t1)
        trades = self.db.query("SELECT tick, symbol, quantity, price, reason FROM trades "
                               "WHERE portfolio_id=? AND tick>? AND tick<=?", (portfolio_id, t0, t1))
        pnl = {}
        for s, q in pos0.items():
            pnl[s] = pnl.get(s, 0.0) + q * (p1[s] - p0[s])
        for t in trades:
            pnl[t["symbol"]] = pnl.get(t["symbol"], 0.0) + t["quantity"] * (p1[t["symbol"]] - t["price"])
        attribution = sorted(({"symbol": s, "pnl": _r(v, 2), "pnl_pct_of_nav": _r(v / then["nav"], 5)}
                              for s, v in pnl.items()), key=lambda r: -abs(r["pnl"]))[:5]
        w0 = {s: q * p0[s] / then["nav"] for s, q in pos0.items()}
        w1 = {s: q * p1[s] / now["nav"] for s, q in pos1.items()}
        wchg = sorted(({"symbol": s, "weight_then": _r(w0.get(s, 0.0)), "weight_now": _r(w1.get(s, 0.0)),
                        "change": _r(w1.get(s, 0.0) - w0.get(s, 0.0))} for s in set(w0) | set(w1)),
                      key=lambda r: -abs(r["change"]))[:5]
        s0, s1 = self._sectors_at(portfolio_id, t0), self._sectors_at(portfolio_id, t1)
        sector_changes = sorted(({"sector": s, "then": _r(s0.get(s, 0.0)), "now": _r(s1.get(s, 0.0)),
                                  "change": _r(s1.get(s, 0.0) - s0.get(s, 0.0))} for s in set(s0) | set(s1)),
                                key=lambda r: -abs(r["change"]))
        return {"portfolio_id": portfolio_id, "from_tick": t0, "to_tick": t1, "metric_changes": changes,
                "trades": [{**t, "notional_pct_of_nav": _r(t["quantity"] * t["price"] / then["nav"], 4),
                            "quantity": _r(t["quantity"], 2), "price": _r(t["price"], 2)} for t in trades],
                "pnl_attribution_top5": attribution, "weight_changes_top5": wchg,
                "sector_changes": sector_changes}

    def get_market_context(self, lookback=24):
        lookback = int(lookback)
        movers = self.db.named("top_movers", lookback=lookback)
        df = pd.DataFrame(movers)
        sector_ret = df.groupby("sector")["ret"].mean().sort_values(key=abs, ascending=False)
        t1 = self.db.query("SELECT MAX(tick) t FROM prices")[0]["t"]
        rows = self.db.query("SELECT tick, symbol, price FROM prices WHERE tick>=?", (t1 - 5 * lookback,))
        px = pd.DataFrame(rows).pivot(index="tick", columns="symbol", values="price")
        mkt = px.pct_change().dropna().mean(axis=1)
        recent, prior = mkt.iloc[-lookback:].std(), mkt.iloc[:-lookback].std()
        return {"lookback": lookback,
                "top_movers": [{"symbol": r["symbol"], "sector": r["sector"], "return": _r(r["ret"], 4)}
                               for r in movers[:5]],
                "sector_returns": {k: _r(float(v), 4) for k, v in sector_ret.items()},
                "market_vol_ratio_recent_vs_prior": _r(float(recent / prior), 2) if prior else None}

    def get_similar_alerts(self, portfolio_id=None, feature=None, limit=5):
        rows = self.db.query(
            "SELECT id, tick, portfolio_id, severity, anomaly_score, drivers, summary FROM alerts "
            "WHERE (:pid IS NULL OR portfolio_id=:pid) AND (:f IS NULL OR drivers LIKE '%\"feature\": \"' || :f || '\"%') "
            "ORDER BY tick DESC LIMIT :lim", {"pid": portfolio_id, "f": feature, "lim": int(limit)})
        return {"alerts": [{**r, "drivers": json.loads(r["drivers"] or "[]")} for r in rows]}
