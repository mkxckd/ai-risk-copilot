"""LLM client selection for the risk agent.

RiskAgent drives an Anthropic-style tool-use loop (`client.messages.create(...)` returning content blocks and a
`stop_reason`). Claude is used through the official SDK; OpenAI-compatible models are supported through a small
adapter that translates the same loop to the Chat Completions function-calling format, so the agent code, tools
and evidence trail are identical for both providers.
"""
import json
import os
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4.1-mini"}


def provider_from_env() -> str | None:
    explicit = os.getenv("RISK_LLM_PROVIDER", "").lower()
    if explicit in ("anthropic", "openai"):
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def model_for(provider: str | None) -> str:
    return os.getenv("RISK_LLM_MODEL") or DEFAULT_MODELS.get(provider or "anthropic", DEFAULT_MODELS["anthropic"])


def default_client():
    provider = provider_from_env()
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            return anthropic.Anthropic()
        except Exception:
            return None
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIToolClient(os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
    return None


def _http_post(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    request = Request(url, data=json.dumps(payload).encode(), method="POST",
                      headers=headers | {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"OpenAI request failed: {type(exc).__name__}") from exc


class OpenAIToolClient:
    """Exposes `.messages.create(...)` with Anthropic semantics on top of OpenAI Chat Completions."""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 30, transport=_http_post):
        self.api_key, self.timeout, self.transport = api_key, timeout, transport
        self.url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.messages = SimpleNamespace(create=self.create)

    @staticmethod
    def _tools(specs: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                  "parameters": s["input_schema"]}} for s in specs]

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
            elif m["role"] == "assistant":
                text = "".join(b.text for b in content if b.type == "text") or None
                calls = [{"id": b.id, "type": "function",
                          "function": {"name": b.name, "arguments": json.dumps(b.input)}} for b in content if b.type == "tool_use"]
                out.append({"role": "assistant", "content": text} | ({"tool_calls": calls} if calls else {}))
            else:  # user turn carrying tool results
                out.extend({"role": "tool", "tool_call_id": r["tool_use_id"], "content": r["content"]} for r in content)
        return out

    def create(self, model: str, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 1500, **_):
        payload = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                   "messages": self._messages(system, messages), "tools": self._tools(tools)}
        data = self.transport(self.url, {"Authorization": f"Bearer {self.api_key}"}, payload, self.timeout)
        choice = data["choices"][0]
        message = choice["message"]
        blocks = []
        if message.get("content"):
            blocks.append(SimpleNamespace(type="text", text=message["content"]))
        for call in message.get("tool_calls") or []:
            blocks.append(SimpleNamespace(type="tool_use", id=call["id"], name=call["function"]["name"],
                                          input=json.loads(call["function"]["arguments"] or "{}")))
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
        return SimpleNamespace(stop_reason=stop, content=blocks)
