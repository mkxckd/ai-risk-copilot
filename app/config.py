import os

DB_PATH = os.getenv("RISK_DB_PATH", "risk_copilot.db")
MODEL_PATH = os.getenv("RISK_MODEL_PATH", "models/anomaly.joblib")
SEED = int(os.getenv("RISK_SEED", "7"))
WINDOW = int(os.getenv("RISK_WINDOW", "120"))          # ticks of history used for risk metrics
BASELINE_TICKS = int(os.getenv("RISK_BASELINE_TICKS", "400"))
SIM_INTERVAL = float(os.getenv("RISK_SIM_INTERVAL", "0"))  # seconds between live ticks; 0 = manual only
LLM_MODEL = os.getenv("RISK_LLM_MODEL", "claude-sonnet-5")

TICKS_PER_DAY = 78                     # 5-minute bars in a 6.5h session
TICKS_PER_YEAR = 252 * TICKS_PER_DAY
ALERT_COOLDOWN_TICKS = 20
