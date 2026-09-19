"""Builds a ready-to-serve system: DB, simulated history, trained detector, tools, agent."""
import os
from dataclasses import dataclass

import pandas as pd

from .agent import RiskAgent
from .anomaly import FEATURES, RiskAnomalyDetector
from .config import BASELINE_TICKS, DB_PATH, MODEL_PATH, SEED
from .db import Database
from .engine import RiskEngine
from .tools import RiskTools


@dataclass
class System:
    db: Database
    engine: RiskEngine
    detector: RiskAnomalyDetector
    tools: RiskTools
    agent: RiskAgent


def build_system(db_path=DB_PATH, model_path=MODEL_PATH, baseline_ticks=BASELINE_TICKS,
                 seed=SEED, retrain=False, llm_client=None) -> System:
    db = Database(db_path)
    engine = RiskEngine(db, seed=seed)
    engine.bootstrap()
    detector = None
    if model_path and os.path.exists(model_path) and not retrain:
        detector = RiskAnomalyDetector.load(model_path)
    else:
        if not db.query("SELECT 1 FROM risk_snapshots LIMIT 1"):
            engine.run(baseline_ticks)                      # detector is None -> unscored baseline snapshots
        rows = db.query(f"SELECT portfolio_id, {','.join(FEATURES)} FROM risk_snapshots WHERE anomaly_score IS NULL")
        detector = RiskAnomalyDetector(seed=seed).fit(pd.DataFrame(rows))
        if model_path:
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            detector.save(model_path)
    engine.detector = detector
    tools = RiskTools(db)
    return System(db, engine, detector, tools, RiskAgent(tools, client=llm_client))
