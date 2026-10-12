"""MCP server exposing the risk copilot's read-only tools to external AI agents.

    python -m app.mcp_server        # stdio transport; uses RISK_DB_PATH / RISK_MODEL_PATH like the API

These are the same tools the built-in RiskAgent calls, so an external agent gets the same evidence. None of them
trades, injects events or writes to the database.
"""
from mcp.server.mcpserver import MCPServer

from .bootstrap import build_system
from .service import RiskService

server = MCPServer(
    name="risk-copilot",
    instructions=("Read-only access to simulated trading portfolios: latest risk metrics, history, positions, "
                  "decomposition of recent risk changes, market context and past anomaly alerts. Every number "
                  "you report should come from a tool result. Data is simulated; this is not investment advice."),
)
_service: RiskService | None = None


def use_system(system) -> None:
    """Bind the server to an existing system (used by tests and when embedding the server)."""
    global _service
    _service = RiskService(system)


def service() -> RiskService:
    global _service
    if _service is None:
        _service = RiskService(build_system())
    return _service


def _tool(name: str, **args) -> dict:
    svc = service()
    if "portfolio_id" in args and args["portfolio_id"] is not None and args["portfolio_id"] not in svc.s.engine.portfolio_ids:
        raise ValueError(f"Unknown portfolio_id {args['portfolio_id']!r}; choose from {svc.s.engine.portfolio_ids}")
    result = svc.s.tools.call(name, {k: v for k, v in args.items() if v is not None})
    if "error" in result:
        raise ValueError(result["error"])
    return result


@server.tool()
def list_portfolios() -> list[dict]:
    """List portfolios with their latest NAV, leverage, 1-day VaR and anomaly flag."""
    return service().portfolios()


@server.tool()
def get_latest_snapshot(portfolio_id: str) -> dict:
    """Latest risk snapshot (NAV, leverage, beta, VaR, volatility, concentration, sector exposure) and the latest alert."""
    return _tool("get_latest_snapshot", portfolio_id=portfolio_id)


@server.tool()
def get_risk_history(portfolio_id: str, n: int = 30) -> dict:
    """Recent risk snapshots with 40-tick moving averages, to judge whether current risk is unusual."""
    return _tool("get_risk_history", portfolio_id=portfolio_id, n=max(1, min(n, 500)))


@server.tool()
def get_positions(portfolio_id: str) -> dict:
    """Current positions with market value, weight and sector."""
    return _tool("get_positions", portfolio_id=portfolio_id)


@server.tool()
def explain_change(portfolio_id: str, lookback: int = 24) -> dict:
    """Metric deltas, trades, P&L attribution and weight/sector changes over the last `lookback` 5-minute ticks."""
    return _tool("explain_change", portfolio_id=portfolio_id, lookback=max(1, min(lookback, 200)))


@server.tool()
def get_market_context(lookback: int = 24) -> dict:
    """Top movers, sector returns and recent-vs-prior market volatility ratio."""
    return _tool("get_market_context", lookback=max(2, min(lookback, 200)))


@server.tool()
def get_similar_alerts(portfolio_id: str | None = None, feature: str | None = None, limit: int = 5) -> dict:
    """Past anomaly alerts, optionally filtered by portfolio and driver feature (e.g. top_weight, leverage, vol_ann)."""
    return _tool("get_similar_alerts", portfolio_id=portfolio_id, feature=feature, limit=max(1, min(limit, 50)))


@server.tool()
def explain_portfolio_risk(portfolio_id: str, question: str | None = None) -> dict:
    """Run the copilot's own explainer (LLM with tools if configured, otherwise rule-based) and return its evidence."""
    svc = service()
    if portfolio_id not in svc.s.engine.portfolio_ids:
        raise ValueError(f"Unknown portfolio_id {portfolio_id!r}")
    return svc.explain(portfolio_id, question)


if __name__ == "__main__":
    server.run("stdio")
