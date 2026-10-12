"""OpenAI tool-calling adapter and MCP server tests. No network calls are made."""
import json
import unittest

import anyio
from mcp import Client

from app import mcp_server
from app.agent import RiskAgent
from app.bootstrap import build_system
from app.llm_clients import OpenAIToolClient


class OpenAIAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = build_system(":memory:", model_path=None, baseline_ticks=120, seed=303)

    def test_tool_loop_runs_through_openai_function_calling(self):
        payloads = []

        def transport(url, headers, payload, timeout):
            payloads.append(payload)
            if len(payloads) == 1:
                return {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_latest_snapshot", "arguments": json.dumps({"portfolio_id": "BAL"})}}]}}]}
            return {"choices": [{"finish_reason": "stop", "message": {"content": "BAL risk is stable."}}]}

        client = OpenAIToolClient("test-key", transport=transport)
        out = RiskAgent(self.s.tools, client=client, model="gpt-test").explain("BAL")
        self.assertEqual(out["mode"], "llm")
        self.assertEqual(out["explanation"], "BAL risk is stable.")
        self.assertEqual(out["evidence"][0]["tool"], "get_latest_snapshot")
        # Second request must carry the assistant tool call and the matching tool result.
        second = payloads[1]["messages"]
        self.assertEqual(second[0]["role"], "system")
        self.assertEqual(second[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second[3]["role"], "tool")
        self.assertEqual(second[3]["tool_call_id"], "call_1")
        self.assertIn("snapshot", json.loads(second[3]["content"]))
        self.assertEqual({t["function"]["name"] for t in payloads[0]["tools"]} >= {"explain_change"}, True)

    def test_openai_failure_falls_back_to_rule_based(self):
        def transport(*args):
            raise RuntimeError("OpenAI request failed: URLError")

        out = RiskAgent(self.s.tools, client=OpenAIToolClient("k", transport=transport)).explain("BAL")
        self.assertEqual(out["mode"], "rule_based_after_llm_error")


class MCPServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mcp_server.use_system(build_system(":memory:", model_path=None, baseline_ticks=120, seed=404))

    def call(self, tool, args):
        async def run():
            async with Client(mcp_server.server) as client:
                names = {t.name for t in (await client.list_tools()).tools}
                result = await client.call_tool(tool, args)
                return names, result
        return anyio.run(run)

    def test_lists_read_only_tools(self):
        names, _ = self.call("list_portfolios", {})
        self.assertEqual(names, {"list_portfolios", "get_latest_snapshot", "get_risk_history", "get_positions",
                                 "explain_change", "get_market_context", "get_similar_alerts", "explain_portfolio_risk"})

    def test_snapshot_tool_returns_metrics(self):
        _, result = self.call("get_latest_snapshot", {"portfolio_id": "TECH"})
        data = json.loads(result.content[0].text)
        self.assertIn("var_95", data["snapshot"])

    def test_unknown_portfolio_is_an_error_not_a_crash(self):
        _, result = self.call("get_positions", {"portfolio_id": "NOPE"})
        self.assertTrue(result.is_error)


if __name__ == "__main__":
    unittest.main()
