"""Create the database + trained model ahead of time: python -m scripts.seed"""
from app.bootstrap import build_system
from app.config import BASELINE_TICKS, DB_PATH, MODEL_PATH

if __name__ == "__main__":
    s = build_system(retrain=True)
    n = s.db.query("SELECT COUNT(*) c FROM risk_snapshots")[0]["c"]
    print(f"db={DB_PATH} model={MODEL_PATH} tick={s.engine.tick} snapshots={n} baseline_ticks={BASELINE_TICKS}")
