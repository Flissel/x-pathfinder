"""OPC-UA bridge stub for real LPBF machines.

This is a placeholder for the asyncua-based OpcUaMachine. It documents the
expected node ID layout per vendor adapter; the real implementation needs
the `asyncua` extra installed (pip install -e ".[hardware]") and a vendor
namespace map under hardware/adapters/{slm,eos,trumpf}.py.

It deliberately does NOT inherit from Machine — calling code must either
install the optional `hardware` extra or use MockMachine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpcUaConfig:
    """Connection + node-id config for OPC-UA bridge.

    Use this as the spec hand-off for vendor adapters. Real implementations
    populate `namespace_map` from `hardware/adapters/<vendor>.py`.
    """

    endpoint: str                      # e.g. "opc.tcp://machine.local:4840"
    namespace_map: dict[str, str]      # logical name -> node-id
    auth_token_env: str = "LASER_SIM_OPCUA_TOKEN"


def opcua_available() -> bool:
    try:
        import asyncua  # noqa: F401
    except ImportError:
        return False
    return True
