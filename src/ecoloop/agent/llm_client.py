"""Ollama tool-calling agent loop, executed via the MCP client.

One :meth:`AgentLoop.decide` call runs a bounded conversation:

* system prompt + few-shot + a user message built from compact summaries,
* ``ollama.chat(model, messages, tools=<mcp tools>)`` with a hard timeout,
* if the model returns tool_calls, each is executed via the MCP client and the
  result appended as a tool message, then we loop,
* termination on a final JSON decision or a max-iteration cap.

Robustness: the whole call is wrapped with a hard timeout (tenacity) and broad
exception handling. On any failure it returns ``(None, meta)`` and the caller
falls back to the last known-good policy.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMCfg
from . import prompts
from .mcp_client import MCPClient

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class AgentLoop:
    def __init__(self, cfg: LLMCfg, mcp_client: MCPClient, logger=None):
        self.cfg = cfg
        self.mcp = mcp_client
        self.tools = mcp_client.list_tools()
        self._log = logger or _default_logger()

    # ------------------------------------------------------------------ #
    def decide(
        self, summary: dict, targets: dict, correction_note: Optional[str] = None
    ) -> Tuple[Optional[dict], Dict[str, Any]]:
        """Run the agent loop; return (decision_dict_or_None, meta)."""
        import ollama

        messages: List[dict] = [
            {"role": "system", "content": prompts.system_prompt()},
            *prompts.FEW_SHOT,
            {"role": "user", "content": prompts.build_user_message(summary, targets, correction_note)},
        ]
        tool_calls_made: List[str] = []
        t0 = time.time()
        client = ollama.Client(host=self.cfg.host, timeout=self.cfg.timeout_s)

        try:
            for _ in range(self.cfg.max_tool_iters):
                resp = self._chat(client, messages)
                msg = resp["message"] if isinstance(resp, dict) else resp.message
                msg_dict = _as_message_dict(msg)
                messages.append(msg_dict)

                calls = msg_dict.get("tool_calls") or []
                if calls:
                    for call in calls:
                        name, args = _extract_call(call)
                        tool_calls_made.append(name)
                        result = self._safe_call(name, args)
                        messages.append(
                            {
                                "role": "tool",
                                "name": name,
                                "content": json.dumps(result, default=str),
                            }
                        )
                    continue  # let the model react to tool results

                # No tool calls -> expect a final decision in the content.
                decision = _parse_decision(msg_dict.get("content", ""))
                if decision is not None:
                    return decision, self._meta(tool_calls_made, t0, ok=True)
                # Nudge once for strict JSON if content wasn't parseable.
                messages.append(
                    {
                        "role": "user",
                        "content": "Emit ONLY the final decision JSON now, no prose.",
                    }
                )
            # Iteration cap hit: try to parse whatever the last message held.
            last = messages[-1].get("content", "") if messages else ""
            decision = _parse_decision(last)
            return decision, self._meta(tool_calls_made, t0, ok=decision is not None,
                                        note="max_iters reached")
        except Exception as exc:  # noqa: BLE001
            self._log.error(f"agent loop error: {exc!r}")
            return None, self._meta(tool_calls_made, t0, ok=False, note=repr(exc))

    # ------------------------------------------------------------------ #
    def _chat(self, client, messages):
        """One ollama.chat call with a hard timeout via tenacity."""
        from tenacity import retry, stop_after_attempt, wait_fixed

        @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5), reraise=True)
        def _do():
            return client.chat(
                model=self.cfg.model,
                messages=messages,
                tools=self.tools,
                options={
                    "num_predict": self.cfg.num_predict,
                    "temperature": self.cfg.temperature,
                },
            )

        return _do()

    def _safe_call(self, name: str, args: dict) -> Any:
        try:
            return self.mcp.call_tool(name, args)
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f"tool '{name}' failed: {exc!r}")
            return {"error": str(exc)}

    def _meta(self, tool_calls, t0, ok, note="") -> Dict[str, Any]:
        return {
            "tool_calls": tool_calls,
            "latency_s": round(time.time() - t0, 3),
            "ok": ok,
            "note": note,
        }


# --------------------------------------------------------------------------- #
# Helpers to normalize ollama's response shapes across versions
# --------------------------------------------------------------------------- #


def _as_message_dict(msg) -> dict:
    if isinstance(msg, dict):
        out = dict(msg)
    else:  # pydantic-style object
        out = {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", "") or "",
        }
        tc = getattr(msg, "tool_calls", None)
        if tc:
            out["tool_calls"] = tc
    out.setdefault("role", "assistant")
    out.setdefault("content", "")
    return out


def _extract_call(call) -> Tuple[str, dict]:
    fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
    if isinstance(fn, dict):
        name = fn.get("name", "")
        args = fn.get("arguments", {})
    else:
        name = getattr(fn, "name", "")
        args = getattr(fn, "arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return name, args or {}


def _parse_decision(content: str) -> Optional[dict]:
    if not content:
        return None
    content = content.strip()
    # Strip markdown fences if present.
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"):] if "{" in content else content
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(content)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) and "setpoints" in obj else None


def _default_logger():
    import logging

    logger = logging.getLogger("ecoloop.agent")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel("INFO")
    logger.propagate = False
    return logger
