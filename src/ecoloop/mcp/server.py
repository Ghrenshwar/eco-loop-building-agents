"""FastMCP server exposing the EcoLoop building-control tools over localhost.

Runs in its own daemon thread so it never blocks the simulation. The agent
connects to it as a real MCP client (see agent/mcp_client.py) and calls the
tools defined in :mod:`.tools`. The server is genuinely inspectable: point any
MCP client (or ``mcp dev``) at ``http://<host>:<port>/mcp`` and you can list and
invoke every tool.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..config import Config
from . import tools as T


def build_server(config: Config):
    """Construct a FastMCP server with all EcoLoop tools registered."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name="ecoloop-building-control",
        host=config.mcp.host,
        port=config.mcp.port,
    )

    # Register each tool. FastMCP reads the function signature + docstring to
    # build the schema and description the LLM sees.
    mcp.tool()(T.get_current_telemetry)
    mcp.tool()(T.get_telemetry_summary)
    mcp.tool()(T.get_targets)
    mcp.tool()(T.set_zone_setpoints)
    mcp.tool()(T.set_ecm_flags)
    mcp.tool()(T.parse_simulation_log)
    mcp.tool()(T.snapshot_current_idf)
    return mcp


class MCPServerThread:
    """Runs a FastMCP server in a background daemon thread."""

    def __init__(self, config: Config):
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._mcp = None
        self._log = _logger(config)

    def start(self) -> None:
        self._mcp = build_server(self.config)
        transport = self.config.mcp.transport

        def _serve():
            try:
                # FastMCP.run signature: run(transport=...). For HTTP transports
                # it uses the host/port passed to the constructor.
                self._mcp.run(transport=transport)
            except Exception as exc:  # noqa: BLE001
                self._log.error(f"MCP server crashed: {exc!r}")

        self._thread = threading.Thread(
            target=_serve, name="mcp-server", daemon=True
        )
        self._thread.start()
        self._log.info(
            f"MCP server started on {self.config.mcp.url} (transport={transport})"
        )

    def stop(self, timeout: float = 5.0) -> None:
        # FastMCP does not expose a clean programmatic stop for the HTTP server;
        # as a daemon thread it is torn down when the process exits. We join
        # briefly so shutdown is orderly when possible.
        if self._thread and self._thread.is_alive():
            self._log.info("MCP server thread will exit with process (daemon).")


def _logger(cfg: Config):
    import logging

    logger = logging.getLogger("ecoloop.mcp")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(cfg.logging.level.upper())
    logger.propagate = False
    return logger


if __name__ == "__main__":
    # Standalone launch for manual inspection (binds a dummy bus).
    from ..bus.control_state import ControlPolicy, default_policy
    from ..bus.telemetry import TelemetryBuffer
    from ..config import load_config, load_targets

    cfg, tgt = load_config(), load_targets()
    telemetry = TelemetryBuffer(pmv_band=tuple(tgt.comfort.pmv_band))
    control = ControlPolicy(default_policy(cfg.zones))
    T.bind_context(
        T.ToolContext(
            config=cfg,
            targets=tgt,
            telemetry=telemetry,
            control=control,
            err_file=cfg.paths.output_dir / "ai" / "eplusout.err",
            baseline_idf=cfg.paths.idf,
            generated_dir=cfg.paths.generated_dir,
        )
    )
    build_server(cfg).run(transport=cfg.mcp.transport)
