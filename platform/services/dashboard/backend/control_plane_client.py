"""Async httpx wrapper around the control-plane read endpoints.

The dashboard only needs read traffic. The control plane requires a credential with
the ``read`` scope for it, supplied here through ``token_provider``.

The control plane is a single-worker uvicorn app on an NFS-backed cluster. Under
the dashboard's steady polling load it occasionally cannot accept a new
connection within a tight deadline, surfacing as httpx.ConnectTimeout. A single
such blip must not blank an entire dashboard page, so every read retries
transient transport errors with a short backoff before giving up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

# Transport-level failures that are safe to retry on an idempotent GET.
_TRANSIENT_ERRORS = (
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


class ControlPlaneClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
        retries: int = 2,
        backoff: float = 0.25,
        token_provider: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Forgiving connect timeout (the part that fails under load); keep read
        # tight so a genuinely hung control plane still fails fast.
        self._timeout = httpx.Timeout(timeout, connect=max(timeout, 10.0))
        self._transport = transport
        self._retries = retries
        self._backoff = backoff
        # Looked up per request rather than captured once, so a credential changed on the
        # Config page takes effect without a restart.
        self._token_provider = token_provider

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=self._transport,
        )

    async def _auth_headers(self) -> dict[str, str]:
        token = await self._token_provider() if self._token_provider else None
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def _get(self, path: str) -> httpx.Response:
        """GET with retry on transient transport errors (idempotent reads only)."""
        last_exc: Exception | None = None
        headers = await self._auth_headers()
        for attempt in range(self._retries + 1):
            try:
                async with self._client() as c:
                    return await c.get(path, headers=headers)
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                if attempt < self._retries:
                    await asyncio.sleep(self._backoff * (attempt + 1))
        # Exhausted retries — re-raise the last transient error.
        assert last_exc is not None
        raise last_exc

    async def list_model_names(self) -> list[str]:
        r = await self._get("/models")
        r.raise_for_status()
        return [row["model_name"] for row in r.json()]

    async def get_meta(self, name: str) -> dict:
        r = await self._get(f"/models/{name}/meta")
        if r.status_code == 404:
            raise KeyError(name)
        r.raise_for_status()
        return r.json()

    async def get_readme(self, name: str) -> tuple[str, str]:
        r = await self._get(f"/models/{name}/readme")
        r.raise_for_status()
        payload = r.json()
        return payload.get("text", ""), payload.get("sha", "")

    def bundled_image_url(self, name: str, filename: str) -> str:
        """URL the browser can fetch directly. No async; pure string."""
        return f"{self._base_url}/models/{name}/images/{filename}"
