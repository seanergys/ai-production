"""The platform must work with ``seanergys_modelzoo`` exactly as upstream publishes it.

Until 2026-08 the platform ran against a vendored fork that carried platform glue inside the
model library (a step decorator, storage backends, a save/load adapter, a configurator base,
``train_step`` wrappers and a ``backend`` field on datasets). Upstream has none of it, so a
fresh clone could not train. The glue now lives in the Seanergy pack; these tests hold the
line by building and training every registered model against the real upstream classes,
offline, using the sample files upstream ships for its own tests.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MZ = Path(os.environ.get("EXAMLOPS_MODELZOO_DIR") or (_REPO_ROOT / "modelzoo"))
_SAMPLES = _MZ / "tests" / "fixtures" / "sample_data"
if not (_MZ / "seanergys_modelzoo").is_dir() or not _SAMPLES.is_dir():
    pytest.skip(
        "seanergys_modelzoo (with its test fixtures) not present — run `make modelzoo` "
        "or set EXAMLOPS_MODELZOO_DIR to a checkout to run these tests.",
        allow_module_level=True,
    )
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "usecases" / "seanergy"), str(_MZ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from seanergy_framework import sources  # noqa: E402

import pipelines.pipeline_generator as pg  # noqa: E402
from pipelines import usecase  # noqa: E402

_PM100_SAMPLE = _SAMPLES / "sample_pm100.parquet"
_FDATA_SAMPLE = _SAMPLES / "sample_fdata.parquet"

# Everything the Seanergy pack binds from the model library itself. If upstream renames or
# drops one of these, the platform breaks — this is the contract to raise with upstream.
_UPSTREAM_BINDINGS = ("model_base", "model_params", "dataloader", "dataloader_params", "model_task")
# Platform glue: must come from the pack, never from the model library.
_PACK_BINDINGS = ("config_base", "pipeline_step", "get_backend", "framework_adapter")


class _SampleBackend:
    """Stands in for an object store: serves upstream's sample files for any logical path."""

    name = "minio"

    def fetch(self, logical_path: str, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sample = _PM100_SAMPLE if logical_path.startswith("PM100/") else _FDATA_SAMPLE
        dest = cache_dir / logical_path.replace("/", "_")
        if not dest.exists():
            shutil.copy(sample, dest)
        return dest


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Cache under tmp_path and serve sample data, so no test touches the network."""
    monkeypatch.setattr(pg, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sources, "get_backend", lambda name, **_: _SampleBackend())
    return tmp_path


def _pairs() -> list[tuple[str, str]]:
    return [
        (model_name, ds_cls.__name__)
        for model_name, (_, cfg, _) in sorted(pg.MODEL_REGISTRY.items())
        for ds_cls in cfg.SUPPORTED_DATASETS
    ]


def test_framework_bindings_resolve_against_upstream():
    framework = usecase.framework()
    for key in _UPSTREAM_BINDINGS:
        assert framework[key].__module__.startswith("seanergys_modelzoo."), key
    for key in _PACK_BINDINGS:
        assert framework[key].__module__.startswith("seanergy_framework."), key


def test_every_enabled_model_is_registered():
    assert {"JPCP", "MACK", "MCBound"} <= set(pg.MODEL_REGISTRY)
    for model_name, (_, _, tasks) in pg.MODEL_REGISTRY.items():
        assert {"train_step", "evaluate_step"} <= set(tasks), model_name


@pytest.mark.parametrize(("model_name", "dataset_name"), _pairs())
def test_dummy_training_runs_on_upstream_classes(offline, model_name, dataset_name):
    _, cfg, _ = pg.MODEL_REGISTRY[model_name]
    ds_cls = pg._resolve_dataset_cls(cfg, dataset_name)

    model, dataset, loader = cfg.get_train_components(
        ds_cls, split="train", is_dummy=True, backend_name="minio"
    )
    assert len(dataset) > 0

    pg._fit(model, loader)

    features, _ = model._extract_data_from_loader(loader)
    assert len(model.estimator.predict(features)) == len(dataset)


def test_pm100_dummy_sample_is_small_and_local(offline):
    _, cfg, _ = pg.MODEL_REGISTRY["JPCP"]
    ds_cls = pg._resolve_dataset_cls(cfg, "PM100Dataset")
    _, dataset, _ = cfg.get_train_components(
        ds_cls, split="train", is_dummy=True, backend_name="minio"
    )
    assert 0 < len(dataset) <= sources._PM100_DUMMY_ROWS
    assert Path(str(dataset.data_path)).is_relative_to(offline)


def test_zenodo_fdata_split_is_fetched_once_and_reused(tmp_path):
    built: list[dict] = []

    class _RecordingDataset:
        """Mimics upstream's ``download_path`` contract without the network."""

        def __init__(self, download_path: str, **kwargs):
            built.append(kwargs)
            Path(download_path).mkdir(parents=True, exist_ok=True)
            shutil.copy(_FDATA_SAMPLE, Path(download_path) / "f_data.parquet")

    schema = {"input_features": ["embedding"], "output_features": ["pclass"]}
    args = (_RecordingDataset, tmp_path, True)
    first = sources._fdata_from_zenodo(*args, [[("adt", ">=", "2023-12-01")]], ["23_12"], schema)
    second = sources._fdata_from_zenodo(*args, [[("adt", ">=", "2024-02-01")]], ["24_02"], schema)

    # A dummy run ignores the split upstream, so train and validation share one fetch.
    assert first == second
    assert len(built) == 1
    assert Path(first["data_path"]).exists()
    # Already applied while fetching; the filter columns are not guaranteed to be kept.
    # None, not []: pyarrow rejects an empty filter list ("Malformed filters").
    assert first["filters"] is None
    # The copy is the published data: transforms are the caller's to apply, exactly once.
    assert "transform" not in built[0]
    assert not list(tmp_path.glob("_staging_*"))
