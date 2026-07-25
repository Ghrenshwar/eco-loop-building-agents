"""Synchronous MCP client wrapper around the async MCP Python SDK.

The supervisor runs in a plain thread, but the MCP SDK is async. This class
owns a dedicated asyncio event loop on its own thread and exposes blocking
``list_tools`` / ``call_tool`` methods that submit coroutines to it. It also
translates MCP tool schemas into the tool format Ollama expects.

The streamable-http client / ClientSession construction differs slightly across
mcp SDK releases, so the connect path is defensive.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional


class MCPClient:
    def __init__(self, url: str, timeout_s: float = 30.0):
        self.url = url
        self.timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-client-loop")
        self._thread.start()
        self._session = None
        self._stack: Optional[AsyncExitStack] = None
        self._tools_cache: List[dict] = []

    # -- event loop plumbing ---------------------------------------------- #
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=self.timeout_s + 10)

    # -- connection ------------------------------------------------------- #
    async def _aconnect(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._stack = AsyncExitStack()
        transport = await self._stack.enter_async_context(streamablehttp_client(self.url))
        # streamablehttp_client yields (read, write) or (read, write, extra).
        read, write = transport[0], transport[1]
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    def connect(self) -> None:
        self._submit(self._aconnect())

    async def _adisconnect(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    def close(self) -> None:
        try:
            self._submit(self._adisconnect())
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- tools ------------------------------------------------------------ #
    async def _alist_tools(self) -> List[dict]:
        resp = await self._session.list_tools()
        out = []
        for t in resp.tools:
            schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": schema,
                    },
                }
            )
        return out

    def list_tools(self) -> List[dict]:
        """Return tools in the Ollama/OpenAI tool-calling schema format."""
        self._tools_cache = self._submit(self._alist_tools())
        return self._tools_cache

    async def _acall(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = await self._session.call_tool(name, arguments)
        # Prefer structured content when available; else concatenate text parts.
        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured
        parts = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
        joined = "\n".join(parts) if parts else ""
        try:
            return json.loads(joined) if joined else {"ok": True}
        except json.JSONDecodeError:
            return {"text": joined}

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        return self._submit(self._acall(name, arguments or {}))
