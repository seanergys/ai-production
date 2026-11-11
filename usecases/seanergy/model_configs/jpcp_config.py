"""jpcp_config — Python shim for JPCP. Provides transforms only.

All declarative config (datasets, features, lifecycle, serving, prefect) lives in
pipelines/models/jpcp.yaml. This shim provides model-bound callables that require
a live model instance and cannot be expressed in YAML.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELZOO = Path(os.environ.get("EXAMLOPS_MODELZOO_DIR") or _REPO_ROOT / "modelzoo")
for _p in (str(_REPO_ROOT), str(_MODELZOO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from seanergys_modelzoo.datasets.f_data import FDataDataset
from seanergys_modelzoo.datasets.pm100 import PM100Dataset
from seanergys_modelzoo.models.tasks.power_consumption_prediction.jpcp.jpcp_model import (
    JPCP,
    Embedding,
)

# PM100 input columns that are label-encoded into ``<col>_cat`` features.
_PM100_CATEGORICAL = ("num_nodes_req", "user_id")


class JPCPConfiguration:
    """Python shim for JPCP — provides transform callables only.

    Does NOT inherit SeanergysModelConfiguration; that base class requires
    abstract methods now handled entirely by YAMLBackedConfig.
    """

    MODEL_CLASS: ClassVar[type] = JPCP
    SUPPORTED_DATASETS: ClassVar[list] = [PM100Dataset, FDataDataset]

    @classmethod
    def resolve_embedding_type(cls, value: str) -> Embedding:
        return Embedding[value]

    @staticmethod
    def _pm100_label_encoder(model: JPCP):
        """DataFrame → DataFrame step adding the ``*_cat`` input columns for PM100.

        Dataset preprocessing passes a DataFrame, while ``JPCP.transform_embeddings`` takes
        a numpy array, so the encoding is done here — the same way upstream's own
        ``JPCP.get_train_config`` does it. The fitted encoders are stored on the model so
        they travel with it through save/load.
        """

        def _encode(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            encoders = []
            for col in _PM100_CATEGORICAL:
                le = LabelEncoder()
                df[f"{col}_cat"] = le.fit_transform(df[col].astype(str))
                encoders.append(le)
            model.embedder = encoders
            return df

        return _encode

    @classmethod
    def get_transforms(cls, model: JPCP, dataset_cls: type) -> dict:
        if dataset_cls is PM100Dataset:
            return {
                "preprocessing_functions": [cls._pm100_label_encoder(model)],
                "target_transform": lambda f: np.mean(f[0]) / f[1],
            }
        if dataset_cls is FDataDataset:
            return {
                "transform": model.embedding_parsing,
                "target_transform": lambda f: f[0] / f[1],
            }
        return {}
