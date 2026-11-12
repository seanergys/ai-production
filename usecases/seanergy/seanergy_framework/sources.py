"""Dataset sourcing — turn a storage backend into kwargs an upstream dataset understands.

Upstream ``seanergys_modelzoo`` datasets know two sources: a local ``data_path`` or their own
Zenodo URL (``use_zenodo_url=True``). They have no notion of the platform's pluggable backends
(MinIO, dataplane). So the platform resolves the backend *before* constructing the dataset and
hands it a plain local path. That keeps the dataset classes exactly as upstream publishes them.

The pipeline engine calls :func:`resolve_dataset_kwargs` (bound via ``pack.toml`` →
``[framework] dataset_source``) and merges the returned kwargs into the dataset constructor.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .backends import ZenodoBackend, get_backend

# Rows kept per split for a dummy PM100 run. Upstream's own ``is_dummy`` handling for PM100
# ORs an extra date filter onto the split filter, which widens the selection instead of
# shrinking it, so the platform cuts the sample itself.
_PM100_DUMMY_ROWS = 50

_PM100_LOGICAL = "PM100/job_table.parquet"


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest* once; a partial download never masquerades as the file."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https Zenodo record URL
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _fetch_local(backend: Any, logical_path: str, cache_dir: Path, cached_name: str) -> Path:
    """Return a local copy of *logical_path*, whichever backend it comes from."""
    if backend is None or isinstance(backend, ZenodoBackend):
        url = ZenodoBackend().fetch(logical_path, cache_dir)
        return _download(url, cache_dir / cached_name)
    return Path(backend.fetch(logical_path, cache_dir))


def _pm100(backend: Any, cache_dir: Path, is_dummy: bool, filters: list | None) -> dict[str, Any]:
    local = _fetch_local(backend, _PM100_LOGICAL, cache_dir, "job_table_full.parquet")
    if not is_dummy:
        return {"data_path": str(local)}

    import pandas as pd  # noqa: PLC0415

    key = hashlib.sha256(f"{local.stat().st_mtime_ns}|{filters!r}".encode()).hexdigest()[:12]
    sample = cache_dir / f"job_table_dummy_{key}.parquet"
    if not sample.exists():
        df = pd.read_parquet(local, filters=filters or None).head(_PM100_DUMMY_ROWS)
        df.to_parquet(sample, index=False)
    return {"data_path": str(sample)}


def _fdata_file_names(dataset_cls: type, files: Any) -> list[str]:
    """Expand a ``files`` spec (None / 'all' / name / list / 'a:b' range) to file names."""
    available = list(dataset_cls.model_fields["AVAILABLE_FILES"].default)
    if files is None or files == "all":
        return available
    if isinstance(files, str):
        if ":" in files:
            start, end = files.split(":")
            return available[available.index(start) : available.index(end) + 1]
        return [files]
    return list(files)


def _fdata_from_zenodo(
    dataset_cls: type,
    cache_dir: Path,
    is_dummy: bool,
    filters: list | None,
    files: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Load one split from Zenodo once, then reuse the local copy.

    A training flow builds each split several times (train, result fetch, evaluate). Going
    back to Zenodo for every one of them is slow and fails whenever Zenodo does, so the
    selection is materialised once through upstream's own ``download_path`` option and
    served from disk afterwards.
    """
    # Upstream's dummy mode ignores the split (it reads a fixed file, unfiltered, capped at a
    # few hundred rows), so every split of a dummy run shares one copy.
    selection = None if is_dummy else (files, filters)
    key = hashlib.sha256(repr((sorted(schema.items(), key=str), selection)).encode()).hexdigest()[
        :12
    ]
    cached = cache_dir / f"f_data_{'dummy' if is_dummy else 'split'}_{key}.parquet"
    if not cached.exists():
        staging = cache_dir / f"_staging_{key}_{os.getpid()}"
        try:
            # No transforms here: the copy must hold the data as published, so that the
            # caller's transforms are applied exactly once, on the dataset it builds.
            dataset_cls(
                use_zenodo_url=True,
                is_dummy=is_dummy,
                files=files,
                filters=filters,
                download_path=str(staging),
                **schema,
            )
            (staging / "f_data.parquet").replace(cached)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    # The filters were applied while building the copy, and the columns they refer to are
    # not necessarily among the ones kept, so they must not be applied a second time.
    # None, not []: pyarrow rejects an empty filter list as "Malformed filters".
    return {"data_path": str(cached), "filters": None}


def _fdata(
    dataset_cls: type,
    backend: Any,
    cache_dir: Path,
    is_dummy: bool,
    filters: list | None,
    files: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    if backend is None or isinstance(backend, ZenodoBackend):
        return _fdata_from_zenodo(dataset_cls, cache_dir, is_dummy, filters, files, schema)

    # Upstream's dummy mode reads a capped number of rows from whatever it is pointed at, so
    # one file is enough.
    names = _fdata_file_names(dataset_cls, files)[: 1 if is_dummy else None]
    fetched = [Path(backend.fetch(f"FData/{name}.parquet", cache_dir)) for name in names]

    # Upstream loads *every* parquet in a directory, so give it one holding exactly these files.
    key = hashlib.sha256("|".join(sorted(str(p) for p in fetched)).encode()).hexdigest()[:12]
    view = cache_dir / "_resolved" / key
    view.mkdir(parents=True, exist_ok=True)
    for src in fetched:
        link = view / src.name
        if not link.exists():
            link.symlink_to(src.resolve())
    return {"data_path": str(view)}


def resolve_dataset_kwargs(
    dataset_cls: type,
    *,
    backend_name: str | None,
    cache_dir: Path | None,
    is_dummy: bool = False,
    filters: list | None = None,
    files: Any = None,
    schema: dict[str, Any] | None = None,
    store_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the constructor kwargs that point *dataset_cls* at its data.

    Args:
        dataset_cls: Upstream dataset class being built.
        backend_name: ``"zenodo"`` / ``"minio"`` / ``"dataplane"`` / ``None`` (= zenodo).
        cache_dir: Where downloads are cached. ``None`` means "no local cache", in which case
            the dataset's own Zenodo path is the only option.
        is_dummy: Whether this is a dummy (smoke) run.
        filters: The split's row filters in pyarrow DNF form.
        files: The split's file selection, for multi-file datasets.
        schema: The dataset's ``input_features`` / ``output_features`` / ``columns`` kwargs.
        store_kwargs: Credentials/endpoint overrides for object-store backends.
    """
    name = (backend_name or "zenodo").lower()
    backend = get_backend(name, **(store_kwargs or {}))
    ds_name = dataset_cls.__name__

    if cache_dir is None:
        if name != "zenodo":
            raise ValueError(
                f"{ds_name}: backend {name!r} needs a cache_dir in the model YAML to download into."
            )
        return {"use_zenodo_url": True}

    if ds_name == "PM100Dataset":
        return _pm100(backend, cache_dir, is_dummy, filters)
    if ds_name == "FDataDataset":
        return _fdata(dataset_cls, backend, cache_dir, is_dummy, filters, files, schema or {})

    if name != "zenodo":
        raise ValueError(
            f"{ds_name}: no source mapping for backend {name!r}. "
            "Add one in usecases/seanergy/seanergy_framework/sources.py."
        )
    return {"use_zenodo_url": True}


__all__ = ["resolve_dataset_kwargs"]
