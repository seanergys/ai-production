"""duration_prediction_bull_config — Python shim for DurationPredictionBull. Transforms only.

All declarative config lives in models/duration_prediction_bull.yaml.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import ClassVar

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELZOO = Path(os.environ.get("EXAMLOPS_MODELZOO_DIR") or _REPO_ROOT / "modelzoo")
for _p in (str(_REPO_ROOT), str(_MODELZOO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from seanergys_modelzoo.datasets.f_data import FDataDataset
from seanergys_modelzoo.models.tasks.duration_prediction.duration_prediction_bull.models.duration_prediction_bull import (  # noqa: E501
    DurationPredictionBull,
)


class DurationPredictionBullConfiguration:
    """Python shim for DurationPredictionBull — provides transform callables only.

    The model trains from the dataset's DataFrame (embedding, cnumr, nnumr, elpl and the
    ``duration`` target), so the only per-item transform is upstream's own flattening of
    one sample into [embedding..., cnumr, nnumr, elpl].
    """

    MODEL_CLASS: ClassVar[type] = DurationPredictionBull
    SUPPORTED_DATASETS: ClassVar[list] = [FDataDataset]

    @classmethod
    def get_transforms(cls, model: DurationPredictionBull, dataset_cls: type) -> dict:
        return {"transform": model.flatten_input_sample}
