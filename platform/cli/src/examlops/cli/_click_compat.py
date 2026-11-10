"""The Click classes the installed Typer actually builds its command tree from.

Typer 0.26+ bundles a private copy of Click (``typer._click``) and no longer uses the standalone
``click`` package at all, so ``click.Context`` / ``click.Command`` / ``click.UsageError`` are
*different classes* from the ones Typer passes around and raises. Code that annotated with them
failed mypy, and code that caught ``click.UsageError`` silently caught nothing.

Older Typer (the project allows ``typer>=0.12``) still builds on standalone Click, so fall back
to it there. Import these names from here rather than from ``click`` whenever they describe a
Typer-built object.
"""

from __future__ import annotations

try:  # Typer >= 0.26: the bundled Click is the one in use.
    from typer._click.core import Command, Context
    from typer._click.exceptions import UsageError
except ImportError:  # pragma: no cover - older Typer builds on the standalone package
    from click import Command, Context, UsageError  # type: ignore[assignment]

__all__ = ["Command", "Context", "UsageError"]
