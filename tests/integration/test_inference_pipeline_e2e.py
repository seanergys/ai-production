"""
End-to-end test for the inference pipeline.

Starts a minimal Ray + Serve cluster, deploys all three pipeline stages,
serves a real threading HTTP server as the downstream MultiModelServer mock,
and verifies the full round-trip from HTTP POST -> FeatureTransformer ->
ModelRouter -> response.

Design note
-----------
respx intercepts httpx at the transport level *in-process only*.  Ray actors
run in separate worker processes, so respx cannot intercept their outbound
calls.  Instead we start a lightweight ``http.server``-based mock in a
background thread before Ray initialises.  Every Ray worker process can reach
it over TCP at ``http://127.0.0.1:<_MOCK_PORT>``.

CPU budget
----------
The pipeline deploys three deployments (ModelRouter, FeatureTransformer,
InferencePipelineIngress), each with ``num_replicas=1``.  Ray allocates 1 CPU
per replica, so we need at least 3 CPUs.  We request 4 to leave headroom.
"""

import importlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest
import ray
from ray import serve

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
_MOCK_PORT = 19210  # threading HTTP mock for the downstream MultiModelServer
_SERVE_PORT = 19111  # Ray Serve ingress

_MOCK_URL = f"http://127.0.0.1:{_MOCK_PORT}"

# ---------------------------------------------------------------------------
# Shared mock-server state (written by tests, read by server handler)
# ---------------------------------------------------------------------------
_responses: dict[str, tuple[dict, int]] = {}
_responses_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Simple threading HTTP mock server
# ---------------------------------------------------------------------------
class _MockHandler(BaseHTTPRequestHandler):
    """Handle POST requests and return pre-configured JSON responses."""

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_len)  # consume body; we don't need it

        path = self.path
        with _responses_lock:
            if path in _responses:
                resp_data, status = _responses[path]
            else:
                resp_data, status = {"detail": "not found"}, 404

        body = json.dumps(resp_data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:  # suppress access logs
        return


# ---------------------------------------------------------------------------
# Ray start-up
# ---------------------------------------------------------------------------
# Ray gives a new node a fixed 30 s to register its raylet with the GCS
# (hard-coded in ray/_private/node.py, not configurable). On a loaded CI runner
# that window can pass before the raylet is up, and ray.init() raises "The
# current node timed out during startup" before any test runs.
_RAY_INIT_ATTEMPTS = 3
_RAY_LOG_FILES = ("raylet.err", "raylet.out", "gcs_server.err", "gcs_server.out")


def _ray_startup_logs(tail_lines: int = 40) -> str:
    """Tails of the raylet/GCS logs of the latest Ray session, for the failure message."""
    logs_dir = Path("/tmp/ray/session_latest/logs")
    parts = []
    for name in _RAY_LOG_FILES:
        path = logs_dir / name
        if path.is_file():
            lines = path.read_text(errors="replace").splitlines()[-tail_lines:]
            parts.append(f"--- {path} (last {len(lines)} lines) ---\n" + "\n".join(lines))
    return "\n".join(parts) or f"(no Ray logs found under {logs_dir})"


def _start_ray() -> None:
    """Start a local Ray node, retrying when it times out during start-up.

    The dashboard is not started: these tests never use it, and it is the
    heaviest of the head-node processes, competing with the raylet for CPU in
    exactly the window that times out.
    """
    for attempt in range(1, _RAY_INIT_ATTEMPTS + 1):
        try:
            ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
            return
        except Exception as exc:  # noqa: BLE001 - Ray raises a bare Exception here
            if "timed out during startup" not in str(exc):
                raise
            logs = _ray_startup_logs()
            ray.shutdown()
            if attempt == _RAY_INIT_ATTEMPTS:
                raise RuntimeError(
                    f"Ray did not start after {_RAY_INIT_ATTEMPTS} attempts: {exc}\n{logs}"
                ) from exc
            print(f"[ray-init] attempt {attempt} timed out; retrying\n{logs}")
            time.sleep(5)


# ---------------------------------------------------------------------------
# Module-scoped fixture: mock server + Ray Serve cluster
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline_url():
    """Start mock HTTP server + Ray Serve, yield ingress base URL, tear down."""
    # 1. Start the mock HTTP server in a background daemon thread.
    mock_server = HTTPServer(("127.0.0.1", _MOCK_PORT), _MockHandler)
    t = threading.Thread(target=mock_server.serve_forever, daemon=True)
    t.start()

    # 2. Tell the inference pipeline to forward requests to our mock.
    os.environ["RAY_SERVE_URL"] = _MOCK_URL

    # 3. Start a fresh local Ray cluster (4 CPUs covers the 3 deployments).
    _start_ray()

    # 4. Start Ray Serve on _SERVE_PORT.
    serve.start(http_options={"host": "127.0.0.1", "port": _SERVE_PORT})

    # 5. Reload the pipeline module *after* setting RAY_SERVE_URL so the
    #    module-level constant _RAY_SERVE_URL picks up the mock URL.
    import serving.inference_pipeline.app as pipeline_module

    importlib.reload(pipeline_module)

    # 6. Deploy the pipeline graph.
    serve.run(
        pipeline_module.pipeline_app,
        route_prefix="/infer-pipeline",
        name="pipeline",
    )

    base = f"http://127.0.0.1:{_SERVE_PORT}/infer-pipeline"
    yield base

    # Tear-down
    serve.shutdown()
    ray.shutdown()
    mock_server.shutdown()
    os.environ.pop("RAY_SERVE_URL", None)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
_GOOD_BODY: dict = {
    "job_id": "test-job-1",
    "model_name": "JPCP",
    "alias": "Production",
    "embedding": [0.1] * 384,
    "num_nodes": 4,
    "user_id": "u001",
}

_MOCK_PREDICTION: dict = {
    "prediction": 142.7,
    "model_version": "5",
    "run_id": "abc123",
    "alias": "Production",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_valid_job_returns_prediction(pipeline_url: str) -> None:
    with _responses_lock:
        _responses.clear()
        # ModelRouter._resolve lowercases model_name (MLflow registry uses
        # lowercase names), so the pipeline requests /predict/jpcp, not /JPCP.
        _responses["/predict/jpcp"] = (_MOCK_PREDICTION, 200)

    r = httpx.post(f"{pipeline_url}/infer", json=_GOOD_BODY, timeout=30)

    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] == 142.7
    assert data["model_version"] == "5"
    assert data["run_id"] == "abc123"


def test_direct_features_reach_the_model_server(pipeline_url: str) -> None:
    """A PM100-trained model's own inputs pass all three stages without an embedding."""
    with _responses_lock:
        _responses.clear()
        _responses["/predict/jpcp_pm100"] = (_MOCK_PREDICTION, 200)

    body = {
        "model_name": "JPCP_PM100",
        "alias": "Staging",
        "features": {"num_nodes_req_cat": 4, "user_id_cat": 12},
    }
    r = httpx.post(f"{pipeline_url}/infer", json=body, timeout=30)

    assert r.status_code == 200
    assert r.json()["prediction"] == 142.7


def test_non_numeric_direct_features_return_422(pipeline_url: str) -> None:
    body = {"model_name": "JPCP_PM100", "features": {"user_id": "smoke-user"}}

    r = httpx.post(f"{pipeline_url}/infer", json=body, timeout=30)

    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_missing_embedding_returns_422(pipeline_url: str) -> None:
    body = {k: v for k, v in _GOOD_BODY.items() if k != "embedding"}

    r = httpx.post(f"{pipeline_url}/infer", json=body, timeout=30)

    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_missing_num_nodes_returns_422(pipeline_url: str) -> None:
    body = {k: v for k, v in _GOOD_BODY.items() if k != "num_nodes"}

    r = httpx.post(f"{pipeline_url}/infer", json=body, timeout=30)

    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_wrong_embedding_size_returns_422(pipeline_url: str) -> None:
    body = {**_GOOD_BODY, "embedding": [0.1] * 100}

    r = httpx.post(f"{pipeline_url}/infer", json=body, timeout=30)

    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_model_not_found_returns_404(pipeline_url: str) -> None:
    body = {**_GOOD_BODY, "model_name": "UNKNOWN"}
    # The mock server returns 404 for any path not in _responses.
    # Clear all prior entries so no stale JPCP or other path leaks through.
    with _responses_lock:
        _responses.clear()

    r = httpx.post(f"{pipeline_url}/infer", json=body, timeout=30)

    assert r.status_code == 404
    assert r.json()["error"] == "model_not_found"
