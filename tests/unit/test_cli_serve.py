from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parents[2] / "platform" / "cli" / "src"))
from examlops.cli.main import app

runner = CliRunner()


def test_serve_reload_all():
    with patch(
        "examlops.cli.commands.serve._client.post", return_value={"reloaded": ["JPCP"], "count": 1}
    ):
        result = runner.invoke(app, ["serve", "reload"])
    assert result.exit_code == 0


def test_serve_reload_one():
    with patch(
        "examlops.cli.commands.serve._client.post", return_value={"reloaded": ["JPCP"], "count": 1}
    ) as mock:
        runner.invoke(app, ["serve", "reload", "--model", "JPCP"])
    assert "/reload/JPCP" in mock.call_args[0][0]


def test_serve_check():
    with patch(
        "examlops.cli.commands.serve._client.get", return_value={"status": "ok", "models": ["JPCP"]}
    ):
        result = runner.invoke(app, ["serve", "check"])
    assert result.exit_code == 0


def test_serve_infer_check_posts_valid_pipeline_payload():
    fake = {"prediction": 42.0, "model_name": "jpcp", "model_version": "3"}
    with patch("examlops.cli.commands.serve._client.post", return_value=fake) as mock_post:
        result = runner.invoke(app, ["serve", "infer-check"])
    assert result.exit_code == 0
    url, body = mock_post.call_args[0]
    assert url.endswith("/infer-pipeline/infer")
    assert body["model_name"] == "JPCP"
    assert body["alias"] == "Production"
    assert body["num_nodes"] == 4
    assert body["user_id"] == "smoke"
    assert len(body["embedding"]) == 384


def test_serve_benchmark_runs_dummy_client():
    with patch("subprocess.run") as mock_run:
        result = runner.invoke(app, ["serve", "benchmark", "--requests", "25"])
    assert result.exit_code == 0
    cmd = mock_run.call_args[0][0]
    assert "platform/clients/dummy_client.py" in cmd
    assert "--benchmark" in cmd
    assert "25" in cmd


def test_serve_models_shows_the_fields_ray_serve_actually_returns():
    """Ray Serve's /models answers with model_name / model_version, not name / version."""
    ray_answer = [
        {
            "model_name": "jpcp_pm100",
            "alias": "Staging",
            "model_version": "4",
            "run_id": "r1",
            "status": "ok",
            "project": None,
        }
    ]
    with patch("examlops.cli.commands.serve._client.get", return_value=ray_answer):
        result = runner.invoke(app, ["serve", "models"])
    assert result.exit_code == 0, result.output
    assert "jpcp_pm100" in result.output
    assert "Staging" in result.output
    assert "4" in result.output
