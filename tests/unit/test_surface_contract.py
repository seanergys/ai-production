"""Cross-surface contract harness (enterprise-readiness Phase 4, items 4.5/4.8).

The audit's cohesion risk: the SAME logical operation is re-coded in the SDK, the MCP tools, the
agent, and the dashboard — and can silently diverge. This harness boots the surfaces against one
ephemeral data plane and asserts they return **identical** results for shared operations, so a change
that breaks parity fails CI. It also pins the canonical result-envelope contract (item 4.6) that every
agent-callable surface must speak.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "platform" / "cli" / "src"))


@pytest.fixture
def active_cluster(tmp_path, monkeypatch):
    """One ephemeral ACTIVE cluster the placement surfaces can both see."""
    monkeypatch.setenv("PLATFORM_DB", str(tmp_path / "platform.db"))
    monkeypatch.setenv("EXAMLOPS_HPC_REGISTRY", str(tmp_path / "clusters.yaml"))
    import examlops.platform_db as pdb
    from examlops.hpc_registry import register_pending

    pdb.init_db()
    register_pending("lxp", "flux", host="lxp-login")
    pdb.set_cluster_state("lxp", "ACTIVE", approved_by="test")
    pdb.record_node_snapshot(
        "lxp", "flux", [{"name": "n0", "gpus": 8, "state": "idle", "cpus": 64}]
    )
    return pdb
