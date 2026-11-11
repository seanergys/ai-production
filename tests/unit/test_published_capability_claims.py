"""Guard the capability claim the dashboard publishes.

* **Dashboard** — the ``/me`` capability list drives which affordances the UI renders, but the
  BFF is the sole enforcement point (F15). An unguarded mutating route is a control the UI
  believes is admin-only and the server lets anyone call.

The check fails in *both* directions: an unguarded route is a failure, and so is leaving a route
on the deliberately-viewer-writable allow-list after it grows a guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# ── dashboard: every mutating route is guarded ────────────────────────────────

_ROUTERS = REPO / "platform" / "services" / "dashboard" / "backend" / "routers"
_MUTATING_VERBS = {"post", "put", "patch", "delete"}

#: Routes a *viewer* is deliberately allowed to call. Each is a write that belongs to the
#: viewer's own session or is read-shaped despite being a POST. Anything not listed must be
#: admin- or capability-guarded.
_VIEWER_WRITABLE = {
    ("auth.py", "login"),  # pre-auth by definition
    ("auth.py", "logout"),
    ("alerts.py", "ack"),  # operational acknowledgement, recorded with the actor
    ("collab.py", "add_comment"),  # F22 collaboration is a viewer affordance
    ("collab.py", "create_snapshot"),
    ("models.py", "predict"),  # inference: a POST because it carries a body, not a write
    ("selfobs.py", "ui_action"),  # F24 telemetry of the caller's own UI action
}


def _guard_helpers(tree: ast.Module, src: str) -> set[str]:
    """Module-level functions that themselves perform a capability/role check."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            seg = ast.get_source_segment(src, node) or ""
            if "deny_reason" in seg or "require_capability" in seg or "HTTP_403" in seg:
                names.add(node.name)
    return names


def _mutating_routes() -> list[tuple[str, str, bool]]:
    out: list[tuple[str, str, bool]] = []
    for path in sorted(_ROUTERS.glob("*.py")):
        src = path.read_text()
        tree = ast.parse(src)
        helpers = _guard_helpers(tree, src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            verbs = [
                d
                for d in node.decorator_list
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr in _MUTATING_VERBS
            ]
            if not verbs:
                continue
            seg = ast.get_source_segment(src, node) or ""
            guarded = (
                "require_capability" in seg
                or "Depends(_admin" in seg  # module-level `_admin = require_role("admin")`
                or 'require_role("admin")' in seg
                or any(f"{h}(" in seg for h in helpers)
            )
            out.append((path.name, node.name, guarded))
    return out


@pytest.mark.skipif(not _ROUTERS.is_dir(), reason="dashboard backend not present")
def test_every_mutating_dashboard_route_is_guarded() -> None:
    routes = _mutating_routes()
    assert routes, "found no mutating routes — the scan itself is broken"
    unguarded = {(f, n) for f, n, ok in routes if not ok}
    surprises = sorted(unguarded - _VIEWER_WRITABLE)
    assert not surprises, (
        "mutating dashboard routes with no capability or admin guard: "
        f"{surprises}. Guard them, or add them to _VIEWER_WRITABLE with the reason."
    )


@pytest.mark.skipif(not _ROUTERS.is_dir(), reason="dashboard backend not present")
def test_viewer_writable_allowlist_has_no_stale_entries() -> None:
    """The other direction: a route that grew a guard must leave the allow-list."""
    routes = _mutating_routes()
    unguarded = {(f, n) for f, n, ok in routes if not ok}
    known = {(f, n) for f, n, _ in routes}
    stale = sorted(e for e in _VIEWER_WRITABLE if e in known and e not in unguarded)
    assert not stale, f"these are guarded now and should leave _VIEWER_WRITABLE: {stale}"
    gone = sorted(e for e in _VIEWER_WRITABLE if e not in known)
    assert not gone, f"_VIEWER_WRITABLE names routes that no longer exist: {gone}"
