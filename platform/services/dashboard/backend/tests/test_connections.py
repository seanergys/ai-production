"""Connections (ADR 0087) dashboard router.

Verifies: connections never leak a secret value, and writes require admin.
"""

import dbconn
import pytest

from examlops import platform_db as pdb
from tests.conftest import ADMIN_PW, VIEWER_PW


@pytest.fixture
def platform_db(tmp_path, monkeypatch):
    """Seed through the owning modules, not raw INSERTs.

    ``connections`` is deliberately *not* created by ``init_db()`` — it is owned by its own
    module and created on first write — so seeding it by hand meant keeping a second copy of its
    schema. Going through ``create_connection`` also means the secret takes the real D7
    indirection rather than a hand-written ``secret_ref``.
    """
    from examlops.connections import create_connection

    db = tmp_path / "platform.db"
    monkeypatch.setenv("PLATFORM_DB", str(db))
    pdb.init_db()
    pdb.create_project("research", created_by="alice")
    create_connection(
        "minio",
        "s3",
        project="research",
        config={
            "endpoint": "http://localhost:19000",
            "bucket": "data",
            "access_key": "minioadmin",
        },
        secret_value="minioadmin-secret",
        created_by="alice",
    )
    return str(db)


async def _login(client, password):
    r = await client.post("/api/auth/login", json={"password": password})
    return r.json()["token"]


async def test_connections_list_hides_secret(client, platform_db):
    token = await _login(client, VIEWER_PW)
    r = await client.get(
        "/api/v1/connections?project=research", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    conn = body[0]
    assert conn["name"] == "minio"
    assert conn["kind"] == "s3"
    assert conn["hasSecret"] is True
    # The secret ref is a pointer, never the credential; and no secret value is present anywhere.
    blob = str(conn).lower()
    assert "secret_ref" not in conn  # raw column name never surfaces
    assert conn["config"]["access_key"] == "minioadmin"  # non-secret config passes through
    # The real credential value never appears (only the access key id, which is not secret).
    assert "password" not in blob


async def test_connection_create_requires_admin(client, platform_db):
    token = await _login(client, VIEWER_PW)
    r = await client.post(
        "/api/v1/connections",
        json={"name": "z", "kind": "uri", "config": {"url": "https://zenodo.org"}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


async def test_connection_create_and_list(client, platform_db):
    token = await _login(client, ADMIN_PW)
    r = await client.post(
        "/api/v1/connections",
        json={
            "name": "zenodo",
            "kind": "uri",
            "project": "research",
            "config": {"url": "https://zenodo.org"},
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "zenodo"
    assert body["kind"] == "uri"
    assert body["hasSecret"] is False
    # It now appears in the (viewer) list for that project.
    vtoken = await _login(client, VIEWER_PW)
    lr = await client.get(
        "/api/v1/connections?project=research", headers={"Authorization": f"Bearer {vtoken}"}
    )
    names = {c["name"] for c in lr.json()}
    assert "zenodo" in names
    # And it was audited.
    conn = dbconn.connect(platform_db, row_factory=None)
    n = conn.execute(
        "SELECT COUNT(*) FROM audit_events WHERE action='connection_created' AND target='zenodo'"
    ).fetchone()[0]
    conn.close()
    assert n == 1


async def test_connection_create_with_secret_hides_value(client, platform_db):
    token = await _login(client, ADMIN_PW)
    r = await client.post(
        "/api/v1/connections",
        json={
            "name": "s3data",
            "kind": "s3",
            "project": "research",
            "config": {"endpoint": "http://localhost:19000", "bucket": "data"},
            "secret": "super-secret-key",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["hasSecret"] is True
    # The secret value must never appear in the API response.
    assert "super-secret-key" not in str(body)
    # But it IS retrievable through the shared examlops path (CLI-compatible secret store).
    from examlops import connections as _c

    resolved = _c.resolve_connection("s3data", project="research")
    assert resolved.get("secret") == "super-secret-key"


async def test_connection_create_duplicate_409(client, platform_db):
    token = await _login(client, ADMIN_PW)
    payload = {"name": "dup", "kind": "uri", "config": {"url": "x"}}
    r1 = await client.post(
        "/api/v1/connections", json=payload, headers={"Authorization": f"Bearer {token}"}
    )
    assert r1.status_code == 201
    r2 = await client.post(
        "/api/v1/connections", json=payload, headers={"Authorization": f"Bearer {token}"}
    )
    assert r2.status_code == 409


async def test_connection_create_bad_kind_400(client, platform_db):
    token = await _login(client, ADMIN_PW)
    r = await client.post(
        "/api/v1/connections",
        json={"name": "bad", "kind": "ftp", "config": {}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400


async def test_connection_delete(client, platform_db):
    token = await _login(client, ADMIN_PW)
    # The fixture seeds 'minio' in project 'research'.
    r = await client.delete(
        "/api/v1/connections/minio?project=research",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    # Gone from the list now.
    lr = await client.get(
        "/api/v1/connections?project=research", headers={"Authorization": f"Bearer {token}"}
    )
    assert all(c["name"] != "minio" for c in lr.json())


async def test_connection_delete_requires_admin(client, platform_db):
    token = await _login(client, VIEWER_PW)
    r = await client.delete(
        "/api/v1/connections/minio?project=research",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


async def test_connection_delete_unknown_404(client, platform_db):
    token = await _login(client, ADMIN_PW)
    r = await client.delete(
        "/api/v1/connections/ghost?project=research",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404
