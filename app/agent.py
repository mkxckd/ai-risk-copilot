"""LLM-powered risk explainer. Uses Claude with tool use when an API client is available; otherwise falls
back to a deterministic explainer that runs the same tools. Both return the tool outputs as evidence."""
import json
import os

from .config import LLM_MODEL
from .tools import TOOL_SPECS

SYSTEM_PROMPT = """You are a trading risk analyst copilot. A risk manager asks why a portfolio's risk changed.
Rules:
- Call tools to gather evidence BEFORE answering: at minimum get_latest_snapshot and explain_change; use
  get_market_context to separate market-wide moves from portfolio-specific ones, and get_similar_alerts for precedent.
- Every number you state must come from a tool result. If evidence is missing or ambiguous, say so.
- Distinguish causes: (1) trades/position changes, (2) market volatility regime, (3) sector or price moves,
  (4) drift in weights from P&L. Say which dominates and quantify it.
- Be concise (under 250 words): headline, what changed, why, precedent, and 1-3 risk-mitigation options
  framed as options for a human to evaluate, not instructions to trade.
- Data is simulated; never present output as investment advice."""

MAX_TOOL_CHARS = 12000


def _default_client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


class RiskAgent:
    def __init__(self, tools, client=None, model: str = LLM_MODEL):
        self.tools, self.model = tools, model
        self.client = client if client is not None else _default_client()

    def explain(self, portfolio_id: str, question: str | None = None, max_steps: int = 6) -> dict:
        if self.client is None:
            return self._fallback(portfolio_id, mode="rule_based")
        try:
            return self._llm(portfolio_id, question, max_steps)
        except Exception as exc:
            out = self._fallback(portfolio_id, mode="rule_based_after_llm_error")
            out["llm_error"] = f"{type(exc).__name__}: {exc}"
            return out

    # ---------- LLM path ----------
    def _llm(self, portfolio_id, question, max_steps):
        prompt = (f"Portfolio: {portfolio_id}. "
                  f"{question or 'Explain what changed in its risk recently and why.'}")
        messages, evidence = [{"role": "user", "content": prompt}], []
        for _ in range(max_steps):
            resp = self.client.messages.create(model=self.model, max_tokens=1500, system=SYSTEM_PROMPT,
                                               tools=TOOL_SPECS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                return {"mode": "llm", "model": self.model, "explanation": text, "evidence": evidence}
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = self.tools.call(block.name, dict(block.input))
                    evidence.append({"tool": block.name, "args": dict(block.input), "result": out})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out, default=str)[:MAX_TOOL_CHARS]})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("agent exceeded max tool-use steps")

    # ---------- deterministic path ----------
    def _fallback(self, portfolio_id, mode):
        snap = self.tools.call("get_latest_snapshot", {"portfolio_id": portfolio_id})
        change = self.tools.call("explain_change", {"portfolio_id": portfolio_id})
        market = self.tools.call("get_market_context", {})
        evidence = [{"tool": "get_latest_snapshot", "args": {"portfolio_id": portfolio_id}, "result": snap},
                    {"tool": "explain_change", "args": {"portfolio_id": portfolio_id}, "result": change},
                    {"tool": "get_market_context", "args": {}, "result": market}]
        if "error" in snap or "error" in change:
            return {"mode": mode, "explanation": f"Could not analyse {portfolio_id}: "
                    f"{snap.get('error') or change.get('error')}", "evidence": evidence}

        alert = snap["latest_alert"]
        drivers = (alert or {}).get("drivers", [])
        feature = drivers[0]["feature"] if drivers else None
        similar = self.tools.call("get_similar_alerts", {"feature": feature, "limit": 4}) if feature else {"alerts": []}
        similar["alerts"] = [a for a in similar.get("alerts", []) if not alert or a["id"] != alert["id"]][:3]
        evidence.append({"tool": "get_similar_alerts", "args": {"feature": feature}, "result": similar})

        m = change["metric_changes"]
        causes = []
        big = [t for t in change["trades"] if abs(t["notional_pct_of_nav"]) >= 0.02]
        if big:
            t = max(big, key=lambda t: abs(t["notional_pct_of_nav"]))
            causes.append(f"Position change: {t['reason']} in {t['symbol']} of {t['notional_pct_of_nav']:+.1%} of NAV "
                          f"at tick {t['tick']}.")
        vr = market.get("market_vol_ratio_recent_vs_prior")
        if vr and vr >= 1.5:
            causes.append(f"Market volatility regime: recent market vol is {vr:.1f}x its prior level.")
        for sc, r in list(market.get("sector_returns", {}).items())[:1]:
            expo = next((s["now"] for s in change["sector_changes"] if s["sector"] == sc), 0.0)
            if abs(r) >= 0.01 and abs(expo) >= 0.10:
                causes.append(f"Sector move: {sc} returned {r:+.1%}; portfolio net exposure {expo:+.0%}.")
        if not causes:
            causes.append("No dominant driver found: no large trades, no volatility regime shift, no large "
                          "exposed-sector move. Likely gradual weight drift from P&L.")

        s = snap["snapshot"]
        status = (f"{alert['severity'].upper()} alert at tick {alert['tick']}" if alert and s["is_anomaly"]
                  else "No active anomaly")
        lines = [f"**{portfolio_id}: {status}.** 1-day 95% VaR {s['var_95']:,.0f} "
                 f"({s['var_95_pct']:.2%} of NAV), leverage {s['leverage']:.2f}x, beta {s['beta']:.2f}, "
                 f"top position {s['top_symbol']} at {s['top_weight']:.1%}.",
                 f"\n**Change over ticks {change['from_tick']}-{change['to_tick']}:** "
                 f"VaR {m['var_95']['change_pct']:+.1f}%, vol {m['vol_ann']['change_pct']:+.1f}%, "
                 f"top weight {m['top_weight']['change_pct']:+.1f}%, leverage {m['leverage']['change_pct']:+.1f}%.",
                 "\n**Likely causes:**"] + [f"- {c}" for c in causes]
        top_pnl = change["pnl_attribution_top5"][:2]
        if top_pnl:
            lines.append("- Largest P&L contributors: " + ", ".join(
                f"{p['symbol']} {p['pnl_pct_of_nav']:+.2%} of NAV" for p in top_pnl) + ".")
        if drivers:
            lines.append("\n**Model drivers:** " + "; ".join(
                f"{d['feature']} z={d['z']:+.1f}" for d in drivers) + ".")
        if similar["alerts"]:
            a = similar["alerts"][0]
            lines.append(f"\n**Precedent:** most recent alert driven by {feature}: {a['portfolio_id']} at tick "
                         f"{a['tick']} ({a['severity']}).")
        lines.append("\n**Options to evaluate:** " + self._options(feature, vr) + " (Simulated data; not investment advice.)")
        return {"mode": mode, "explanation": "\n".join(lines), "evidence": evidence}

    @staticmethod
    def _options(feature, vol_ratio):
        if feature in ("top_weight", "hhi"):
            return "trim the concentrated position or add an offsetting hedge; check limits on single-name weight."
        if feature == "leverage":
            return "review gross exposure against the leverage limit; consider reducing gross."
        if feature in ("vol_ann", "var_95_pct", "drawdown") or (vol_ratio and vol_ratio >= 1.5):
            return "re-run stress tests under the elevated-vol regime; consider scaling exposure to a volatility target."
        if feature in ("sector_max", "beta"):
            return "review sector/beta limits; consider hedging the dominant sector or market beta."
        return "monitor; no action indicated by the data."
