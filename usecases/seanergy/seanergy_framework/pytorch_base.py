"""PyTorch base for SeanergysModel — Phase 5.

Concrete PyTorch models subclass ``SeanergysPyTorchModel`` and own their
forward/loss loop in ``train(loader)``. The pipeline picks the right
MLflow flavour automatically because this class declares ``framework =
"pytorch"`` (consumed by ``framework_adapter.adapter_for``).
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field
from seanergys_modelzoo.models.common.seanergys_model import SeanergysModel


class SeanergysPyTorchModel(SeanergysModel):
    """Base class for PyTorch-backed Seanergys models.

    Subclasses are expected to:

    1. Set ``estimator`` to a ``torch.nn.Module`` instance in
       ``model_post_init`` (or via a Pydantic default factory).
    2. Implement ``train(loader)`` to own the optimiser + loss loop.
    3. Implement ``predict(loader)`` as the model base class requires.
    """

    framework: ClassVar[str] = "pytorch"

    estimator: Any = Field(
        default=None,
        description="The torch.nn.Module instance — populated by model_post_init.",
    )
