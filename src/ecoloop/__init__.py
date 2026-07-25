"""EcoLoop Building Agents — autonomous closed-loop building-energy control.

An EnergyPlus digital twin is supervised by a locally-hosted open-source LLM
that reasons over live telemetry and injects HVAC setpoints back into the
running simulation via a real MCP tool server. See docs/ARCHITECTURE.md.
"""

__version__ = "0.1.0"
