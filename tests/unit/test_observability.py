import importlib
from pathlib import Path

from examlops import observability


def test_setup_tracing_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert observability.tracing_enabled() is False
    assert observability.setup_tracing("test-svc") is False


def test_setup_tracing_enabled_configures_provider(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    importlib.reload(observability)
    assert observability.tracing_enabled() is True
    assert observability.setup_tracing("test-svc") is True

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    assert isinstance(trace.get_tracer_provider(), TracerProvider)


def test_ray_metrics_survive_a_disabled_otel_sdk():
    """Ray records its metrics through the OpenTelemetry SDK unless told otherwise.

    The stack sets OTEL_SDK_DISABLED=true by default, which turns that SDK into a no-op:
    ray-serving's :8080/metrics then carried no ray_* series at all, and Prometheus scraped
    an "up" target with nothing in it for as long as its history went back. Ray's own
    metrics path ignores the flag, so ray-serving must select it whenever the SDK is off.
    """
    import yaml

    compose = (
        Path(__file__).resolve().parents[2] / "platform/infra/docker-compose/docker-compose.yml"
    )
    env = yaml.safe_load(compose.read_text())["services"]["ray-serving"]["environment"]
    if "true" in str(env.get("OTEL_SDK_DISABLED", "")).lower():
        assert str(env.get("RAY_enable_open_telemetry")) == "0", env
