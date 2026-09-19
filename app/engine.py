"""Real-time risk engine: advances the market, books trades/events, snapshots risk, scores anomalies."""
import json
import threading
from collections import deque

import numpy as np
import pandas as pd

from .config import ALERT_COOLDOWN_TICKS, SEED, WINDOW
from .risk import compute_risk
from .simulator import UNIVERSE, MarketSimulator, initial_portfolios, tick_ts

PRE_KINDS = {"vol_spike", "sector_crash"}
POST_KINDS = {"concentrate"}
SNAPSHOT_COLS = ("portfolio_id tick ts nav gross net leverage beta var_95 var_99 var_95_pct vol_ann "
                 "drawdown top_weight top_symbol hhi sector_max sector_max_name anomaly_score is_anomaly").split()


class RiskEngine:
    def __init__(self, db, seed: int = SEED, window: int = WINDOW):
        self.db, self.seed, self.window = db, seed, window
        self.instruments = {i.symbol: {"sector": i.sector, "beta": i.beta, "ann_vol": i.ann_vol} for i in UNIVERSE}
        self.symbols = [i.symbol for i in UNIVERSE]
        self.detector = None
        self.tick = 0
        self.events: list[dict] = []
        self.lock = threading.RLock()
        self.hist: deque = deque(maxlen=window + 1)
        self.positions: dict[str, dict] = {}
        self.cash: dict[str, float] = {}
        self.last_alert: dict[str, tuple] = {}
        self.sim = None

    # ---------- setup ----------
    def bootstrap(self):
        with self.lock:
            if self.db.query("SELECT 1 FROM prices LIMIT 1"):
                return self._resume()
            self.sim = MarketSimulator(self.seed)
            self.db.executemany("INSERT INTO instruments VALUES (?,?,?,?)",
                                [(i.symbol, i.sector, i.beta, i.ann_vol) for i in UNIVERSE])
            self._record_prices()
            for p in initial_portfolios(self.sim.prices):
                self.db.execute("INSERT INTO portfolios VALUES (?,?,?,?,?)",
                                (p["portfolio_id"], p["name"], p["strategy"], p["capital"], p["capital"]))
                self.positions[p["portfolio_id"]], self.cash[p["portfolio_id"]] = {}, p["capital"]
                for sym, q in p["positions"].items():   # cash is debited as each initial fill is booked
                    self._book(p["portfolio_id"], sym, q, self.sim.prices[sym], "initial")
            for _ in range(self.window):        # warm-up history so risk metrics exist on tick 1
                self.tick += 1
                self.sim.step()
                self._record_prices()

    def _resume(self):
        last = self.db.query("SELECT MAX(tick) t FROM prices")[0]["t"]
        self.tick = last
        rows = self.db.query("SELECT tick, symbol, price FROM prices WHERE tick > ? ORDER BY tick",
                             (last - self.window - 1,))
        df = pd.DataFrame(rows).pivot(index="tick", columns="symbol", values="price")[self.symbols]
        self.hist.extend(df.to_numpy())
        self.sim = MarketSimulator(self.seed + last, prices=df.iloc[-1].to_dict())
        for r in self.db.query("SELECT * FROM v_positions"):
            self.positions.setdefault(r["portfolio_id"], {})[r["symbol"]] = r["quantity"]
        self.cash = {r["portfolio_id"]: r["cash"] for r in self.db.query("SELECT * FROM portfolios")}

    @property
    def portfolio_ids(self):
        return list(self.positions)

    # ---------- state changes ----------
    def _record_prices(self):
        ts = tick_ts(self.tick)
        self.db.executemany("INSERT INTO prices VALUES (?,?,?,?)",
                            [(self.tick, ts, s, self.sim.prices[s]) for s in self.symbols])
        self.hist.append(np.array([self.sim.prices[s] for s in self.symbols]))

    def _book(self, pid, sym, qty, price, reason, persist_cash=True):
        self.positions[pid][sym] = self.positions[pid].get(sym, 0.0) + qty
        self.cash[pid] -= qty * price
        self.db.execute("INSERT INTO trades (tick, ts, portfolio_id, symbol, quantity, price, reason) "
                        "VALUES (?,?,?,?,?,?,?)", (self.tick, tick_ts(self.tick), pid, sym, qty, price, reason))
        if persist_cash:
            self.db.execute("UPDATE portfolios SET cash=? WHERE portfolio_id=?", (self.cash[pid], pid))

    def _nav(self, pid):
        return self.cash[pid] + sum(q * self.sim.prices[s] for s, q in self.positions[pid].items())

    def schedule(self, kind: str, delay: int = 1, **params):
        if kind not in PRE_KINDS | POST_KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        if kind == "concentrate" and params.get("portfolio_id") not in self.positions:
            raise ValueError("unknown portfolio_id")
        with self.lock:
            self.events.append({"tick": self.tick + max(1, delay), "kind": kind, **params})

    def _apply_events(self, kinds):
        due = [e for e in self.events if e["tick"] == self.tick and e["kind"] in kinds]
        for e in due:
            self.events.remove(e)
            if e["kind"] == "vol_spike":
                self.sim.spike_vol(e.get("mult", 3.0), e.get("duration", 30))
            elif e["kind"] == "sector_crash":
                self.sim.shock_sector(e["sector"], e.get("drift", -0.004), e.get("duration", 10))
            elif e["kind"] == "concentrate":
                pid, sym, target = e["portfolio_id"], e["symbol"], e.get("target_weight", 0.4)
                px = self.sim.prices[sym]
                cur = self.positions[pid].get(sym, 0.0) * px
                self._book(pid, sym, (target * self._nav(pid) - cur) / px, px, "concentration_event")

    # ---------- main loop ----------
    def step(self) -> list[dict]:
        with self.lock:
            self.tick += 1
            self._apply_events(PRE_KINDS)
            self.sim.step()
            self._record_prices()
            self._apply_events(POST_KINDS)
            arr = np.array(self.hist)
            rets = pd.DataFrame(np.diff(arr, axis=0) / arr[:-1], columns=self.symbols)
            risks = {pid: compute_risk(self.positions[pid], self.sim.prices, self.cash[pid], rets, self.instruments)
                     for pid in self.portfolio_ids}
            blank = {"anomaly_score": None, "is_anomaly": None, "severity": "none", "drivers": []}
            scores = {pid: blank for pid in risks}
            if self.detector is not None:
                known = [pid for pid in risks if self.detector.known(pid)]
                scores.update(zip(known, self.detector.score_batch([(pid, risks[pid]) for pid in known])))
            out = []
            for pid, risk in risks.items():
                scored = scores[pid]
                snap = {**risk, "portfolio_id": pid, "tick": self.tick, "ts": tick_ts(self.tick),
                        "anomaly_score": scored["anomaly_score"],
                        "is_anomaly": None if scored["is_anomaly"] is None else int(scored["is_anomaly"])}
                self.db.execute(f"INSERT INTO risk_snapshots ({','.join(SNAPSHOT_COLS)}) "
                                f"VALUES ({','.join('?' * len(SNAPSHOT_COLS))})", [snap[c] for c in SNAPSHOT_COLS])
                self.db.executemany("INSERT INTO sector_exposures VALUES (?,?,?,?)",
                                    [(pid, self.tick, s, w) for s, w in risk["sector_weights"].items()])
                if scored["is_anomaly"]:
                    self._maybe_alert(pid, scored, snap)
                out.append({**snap, **{k: scored[k] for k in ("severity", "drivers")}})
            return out

    def run(self, n: int):
        for _ in range(n):
            self.step()

    def _maybe_alert(self, pid, scored, snap):
        prev = self.last_alert.get(pid)
        feats = {d["feature"] for d in scored["drivers"]}
        # During cooldown only re-alert on escalation (warning -> critical) or a NEW driver feature.
        escalated = prev and prev[1] == "warning" and scored["severity"] == "critical"
        new_driver = prev and bool(feats - prev[2])
        if prev and self.tick - prev[0] < ALERT_COOLDOWN_TICKS and not (escalated or new_driver):
            return
        self.last_alert[pid] = (self.tick, scored["severity"], feats | (prev[2] if prev and self.tick - prev[0] < ALERT_COOLDOWN_TICKS else set()))
        top = "; ".join(f"{d['feature']} {d['value']:.4g} vs baseline {d['baseline']:.4g} (z={d['z']:+.1f})"
                        for d in scored["drivers"][:2]) or "multivariate deviation"
        self.db.execute("INSERT INTO alerts (tick, ts, portfolio_id, severity, anomaly_score, drivers, summary) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (self.tick, snap["ts"], pid, scored["severity"], scored["anomaly_score"],
                         json.dumps(scored["drivers"]), f"{scored['severity'].upper()}: {top}"))
