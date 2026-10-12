"""Instant stress tests: P&L of a portfolio's current positions under market and sector shocks.

Each position's shocked return is   beta_i * market_shock + sector_shock[sector_i]   (the same factor
structure the simulator uses), applied to today's market value. This is a linear, instantaneous shock:
no second-order effects, liquidity costs or re-hedging.
"""
from .simulator import SECTORS

STANDARD_SCENARIOS = [
    {"name": "Market -10%", "market": -0.10, "sectors": {}},
    {"name": "Market -20% (crash)", "market": -0.20, "sectors": {}},
    {"name": "Market +10% (rally)", "market": 0.10, "sectors": {}},
    {"name": "Tech sell-off -15%", "market": 0.0, "sectors": {"Tech": -0.15}},
    {"name": "Energy shock -20%", "market": 0.0, "sectors": {"Energy": -0.20}},
    {"name": "Financials -15%", "market": 0.0, "sectors": {"Financials": -0.15}},
    {"name": "Rotation: Tech -10%, Health +5%", "market": 0.0, "sectors": {"Tech": -0.10, "Health": 0.05}},
]


def validate(scenario: dict) -> dict:
    market = float(scenario.get("market", 0.0))
    sectors = {k: float(v) for k, v in (scenario.get("sectors") or {}).items()}
    unknown = set(sectors) - set(SECTORS)
    if unknown:
        raise ValueError(f"unknown sectors {sorted(unknown)}; choose from {SECTORS}")
    if not -0.9 <= market <= 0.9 or any(not -0.9 <= v <= 0.9 for v in sectors.values()):
        raise ValueError("shocks must be between -0.9 and 0.9 (i.e. -90% to +90%)")
    return {"name": scenario.get("name") or "Custom scenario", "market": market, "sectors": sectors}


def run_stress(tools, db, portfolio_id: str, scenarios: list[dict] | None = None) -> dict:
    book = tools.get_positions(portfolio_id)
    beta = {r["symbol"]: r["beta"] for r in db.query("SELECT symbol, beta FROM instruments")}
    nav = book["nav"]
    results = []
    for raw in scenarios or STANDARD_SCENARIOS:
        sc = validate(raw)
        contributions = []
        for p in book["positions"]:
            shock = beta[p["symbol"]] * sc["market"] + sc["sectors"].get(p["sector"], 0.0)
            contributions.append({"symbol": p["symbol"], "sector": p["sector"],
                                  "pnl": round(p["market_value"] * shock, 2)})
        pnl = sum(c["pnl"] for c in contributions)
        worst = sorted(contributions, key=lambda c: c["pnl"])[:3]
        results.append({**sc, "pnl": round(pnl, 2), "pnl_pct_nav": round(pnl / nav, 5),
                        "worst_contributors": [c | {"pnl_pct_nav": round(c["pnl"] / nav, 5)} for c in worst]})
    return {"portfolio_id": portfolio_id, "tick": book["tick"], "nav": nav, "sectors": SECTORS, "scenarios": results}
