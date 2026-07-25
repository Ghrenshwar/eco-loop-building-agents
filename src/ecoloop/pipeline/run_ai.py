"""AI closed-loop run: MCP server + supervisor thread drive the setpoints.

Same building/weather/period as the baseline; only the control differs. Starts
the FastMCP server, binds the tool context to the live bus, connects the MCP
client, launches the throttled supervisor, then runs EnergyPlus. On completion
(or Ctrl-C) it shuts the threads down cleanly and records the run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..bus.control_state import ControlPolicy, default_policy
from ..bus.telemetry import TelemetryBuffer
from ..config import load_config, load_targets
from ..energyplus.runner import EnergyPlusRunner
from ..logging.recorder import Recorder
from ..mcp import tools as T
from ..mcp.server import MCPServerThread
from ..agent.mcp_client import MCPClient
from ..agent.supervisor import Supervisor
from .bootstrap import prepare_baseline, require_energyplus


def run_ai(smoke: bool = False, force_prepare: bool = False) -> Path:
    cfg, tgt = load_config(), load_targets()
    require_energyplus(cfg)

    model = prepare_baseline(cfg, smoke=smoke, force=force_prepare)
    outdir = cfg.paths.output_dir / "ai"
    outdir.mkdir(parents=True, exist_ok=True)

    telemetry = TelemetryBuffer(pmv_band=tuple(tgt.comfort.pmv_band))
    control = ControlPolicy(default_policy(cfg.zones, heating_sp=21.0, cooling_sp=24.0))
    recorder = Recorder("ai", cfg.paths.output_dir)

    # 1. Bind the MCP tool context to the live bus, then start the server.
    T.bind_context(
        T.ToolContext(
            config=cfg,
            targets=tgt,
            telemetry=telemetry,
            control=control,
            err_file=outdir / "eplusout.err",
            baseline_idf=cfg.paths.idf,
            generated_dir=cfg.paths.generated_dir,
        )
    )
    server = MCPServerThread(cfg)
    server.start()
    time.sleep(1.5)  # let the HTTP server bind before the client connects

    # 2. Connect the MCP client the agent will use.
    mcp_client = MCPClient(cfg.mcp.url, timeout_s=cfg.llm.timeout_s)
    try:
        mcp_client.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"[ai] WARNING: MCP client failed to connect ({exc}); "
              f"supervisor will fall back to last-good policy each cycle.")

    # 3. Start the supervisor (parse_log hook lets it self-correct on log errors).
    supervisor = Supervisor(
        config=cfg,
        targets=tgt,
        telemetry=telemetry,
        control=control,
        mcp_client=mcp_client,
        recorder=recorder,
        parse_log_fn=T.parse_simulation_log,
    )
    supervisor.start()

    # 4. Run EnergyPlus (blocks this thread until the sim completes).
    runner = EnergyPlusRunner(
        config=cfg,
        targets=tgt,
        control_policy=control,
        telemetry=telemetry,
        wired_scheds=model.wired,
        on_timestep=recorder.record_telemetry,
        pace_s=cfg.supervisor.realtime_pace_s_per_step,
    )
    try:
        artifacts = runner.run(model.idf, model.epw, outdir)
    finally:
        # 5. Clean shutdown regardless of how the sim ended.
        supervisor.stop()
        mcp_client.close()
        server.stop()

    written = recorder.flush(sql_source=artifacts.sql_file)
    print(f"[ai] telemetry rows={recorder.n_telemetry} decisions={recorder.n_decisions} "
          f"exit={artifacts.exit_code} outputs={list(written.values())}")
    return outdir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the AI closed-loop simulation.")
    ap.add_argument("--smoke", action="store_true", help="single design day (fast)")
    ap.add_argument("--force-prepare", action="store_true", help="rebuild baseline.idf from example")
    args = ap.parse_args()
    run_ai(smoke=args.smoke, force_prepare=args.force_prepare)


if __name__ == "__main__":
    main()
