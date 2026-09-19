"""Anomaly detection on portfolio risk snapshots.

Each portfolio's features are converted to robust z-scores against *its own* baseline (median / MAD, floored at a materiality level),
so a 3x-levered fund and a long-only fund are compared to their own normal. An IsolationForest is
trained on the pooled z-scores. An alert needs BOTH a low forest score AND at least one feature
beyond Z_GUARD sigma, which keeps the false-positive rate down and gives every alert human-readable drivers.
"""
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES = ["leverage", "beta", "var_95_pct", "top_weight", "hhi", "sector_max", "vol_ann", "drawdown"]
Z_CLIP, Z_GUARD, Z_CRITICAL = 30.0, 4.0, 8.0
# Materiality floors: the smallest move in each feature a risk manager would care about. They stop a
# near-zero baseline (e.g. beta of a beta-neutral book) from turning ordinary weight drift into huge z-scores.
MATERIALITY = {"leverage": 0.05, "beta": 0.05, "var_95_pct": 0.002, "top_weight": 0.01,
               "hhi": 0.005, "sector_max": 0.02, "vol_ann": 0.02, "drawdown": 0.01}
FLOOR = np.array([MATERIALITY[f] for f in FEATURES])


class RiskAnomalyDetector:
    def __init__(self, n_estimators=150, quantile=0.01, seed=7):
        self.n_estimators, self.quantile, self.seed = n_estimators, quantile, seed
        self.model = None
        self.medians: dict = {}
        self.scales: dict = {}
        self.threshold = 0.0

    def _z(self, portfolio_id, x):
        return np.clip((x - self.medians[portfolio_id]) / self.scales[portfolio_id], -Z_CLIP, Z_CLIP)

    def fit(self, df):
        """df: rows of baseline snapshots with portfolio_id + FEATURES columns."""
        zs = []
        for pid, g in df.groupby("portfolio_id"):
            X = g[FEATURES].to_numpy(float)
            med = np.median(X, axis=0)
            mad = 1.4826 * np.median(np.abs(X - med), axis=0)
            self.medians[pid] = med
            self.scales[pid] = np.maximum.reduce([mad, 0.10 * np.abs(med), FLOOR])
            zs.append(self._z(pid, X))
        Z = np.vstack(zs)
        self.model = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(Z)
        d = self.model.decision_function(Z)
        self.threshold = float(min(np.quantile(d, self.quantile), 0.0))
        return self

    def known(self, portfolio_id) -> bool:
        return portfolio_id in self.medians

    def score(self, portfolio_id: str, row: dict) -> dict:
        return self.score_batch([(portfolio_id, row)])[0]

    def score_batch(self, items: list) -> list[dict]:
        """Score several (portfolio_id, feature_row) pairs with ONE forest call (per-call overhead dominates)."""
        X = np.array([[float(row[f]) for f in FEATURES] for _, row in items])
        Z = np.vstack([self._z(pid, x) for (pid, _), x in zip(items, X)])
        d_all = self.model.decision_function(Z)
        out = []
        for (pid, _), x, z, d in zip(items, X, Z, d_all):
            max_z = float(np.abs(z).max())
            is_anom = bool(d < self.threshold and max_z >= Z_GUARD)
            order = np.argsort(-np.abs(z))
            drivers = [{"feature": FEATURES[i], "value": float(x[i]),
                        "baseline": float(self.medians[pid][i]), "z": float(z[i])}
                       for i in order[:3] if abs(z[i]) >= 3.0]
            out.append({
                "anomaly_score": round(-float(d), 4),      # higher = more anomalous
                "is_anomaly": is_anom,
                "severity": ("critical" if max_z >= Z_CRITICAL else "warning") if is_anom else "none",
                "drivers": drivers,
            })
        return out

    def save(self, path):
        joblib.dump({"model": self.model, "medians": self.medians, "scales": self.scales,
                     "threshold": self.threshold, "seed": self.seed}, path)

    @classmethod
    def load(cls, path):
        blob = joblib.load(path)
        det = cls(seed=blob["seed"])
        det.model, det.medians, det.scales, det.threshold = (
            blob["model"], blob["medians"], blob["scales"], blob["threshold"])
        return det
