"""Platform-side framework glue for the Seanergy use-case pack.

These helpers are what the ExaMLOps pipeline engine needs *around* a model — step markers,
dataset storage backends, per-framework save/load, and the configurator base class. They used
to live in a vendored fork of ``seanergys_modelzoo``; upstream modelzoo does not (and should
not) carry them, so they live here and ``pack.toml`` binds them. Upstream is used as published.
"""
