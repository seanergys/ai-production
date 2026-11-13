from __future__ import annotations

import json

import typer

from examlops.cli import _client, _output
from examlops.cli._config import load_config
from examlops.cli._enums import MLflowAlias

_EXAMPLES = (
    "Examples:\n\n"
    '  exa predict JPCP --features "$(python3 -c '
    '\'import json; print(json.dumps({"embedding":[0.1]*384,"num_nodes": 4,"user_id": "smoke"}))\''
    ')"\n\n'
    '  exa predict JPCP --features "$(python3 -c '
    '\'import json; print(json.dumps({"embedding":[0.1]*384,"num_nodes": 4,"user_id": "smoke"}))\''
    ')" --alias Canary\n\n'
    "  # A model trained on another dataset (e.g. PM100): pass its own inputs\n"
    '  exa predict JPCP_PM100 --features \'{"num_nodes_req_cat": 4, "user_id_cat": 12}\''
    " --alias Staging\n\n"
    "  # FData-trained models need a 384-dim embedding and num_nodes. Any --features\n"
    "  # without an embedding is sent to the model unchanged, as its inputs."
)


def predict(
    model: str = typer.Argument(..., help="Model ID (e.g. JPCP)"),
    features: str = typer.Option(..., "--features", "-f", help="JSON feature dict"),
    alias: MLflowAlias | None = typer.Option(None, "--alias", help="MLflow alias"),
    version: str | None = typer.Option(None, "--version", help="Explicit model version"),
):
    """Send an inference request to the Ray Serve inference pipeline."""
    cfg = load_config()
    try:
        feat_dict = json.loads(features)
    except json.JSONDecodeError as e:
        _output.error(f"--features must be valid JSON: {e}")
        return
    if not isinstance(feat_dict, dict):
        _output.error("--features must be a JSON object")
        return
    # FData form: the embedding plus job metadata travel as top-level fields, and the
    # pipeline builds the model input. Anything else is the model's own inputs (e.g. a
    # PM100-trained model's categorical codes), sent as "features" and forwarded unchanged.
    if "embedding" in feat_dict or "features" in feat_dict:
        body: dict = {**feat_dict, "model_name": model}
    else:
        body = {"features": feat_dict, "model_name": model}
    if alias:
        body["alias"] = alias
    if version:
        body["version"] = version
    try:
        result = _client.post(f"{cfg.ray_serve_url}/infer-pipeline/infer", body)
    except _client.ClientError as e:
        _output.error(str(e))
        return
    _output.print_record(result)
