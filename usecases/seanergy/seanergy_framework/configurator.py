"""Configurator base class — binds a model to its datasets via typed parameter bundles.

The parameter bundle types come from upstream ``seanergys_modelzoo``; only the abstract
configurator itself is platform glue. YAML-backed models (the default since Phase 14) do not
subclass this; it remains the base for hand-written and scaffolded configurators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from seanergys_modelzoo.models.common.seanergys_configurator import (
    SeanergysDataloaderParams,
    SeanergysDatasetParams,
    SeanergysModelParams,
)

ParamsBundle = tuple[SeanergysModelParams, SeanergysDatasetParams, SeanergysDataloaderParams]


class SeanergysModelConfiguration(BaseModel, ABC):
    """Abstract configurator: one subclass per concrete model.

    Implement the four ``get_*_params`` classmethods for every dataset the model supports.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ClassVars (not Pydantic fields): subclasses override them at class level and the
    # pipeline engine reads them without instantiating the configurator.
    SUPPORTED_DATASETS: ClassVar[list[Any]] = []
    MODEL_CLASS: ClassVar[Any] = None

    @classmethod
    @abstractmethod
    def get_dummy_params(cls, dataset: Any) -> ParamsBundle:
        """Return the parameter bundle for dummy testing on the given dataset."""

    @classmethod
    @abstractmethod
    def get_training_params(cls, dataset: Any) -> ParamsBundle:
        """Return the parameter bundle for training on the given dataset."""

    @classmethod
    @abstractmethod
    def get_testing_params(cls, dataset: Any) -> ParamsBundle:
        """Return the parameter bundle for testing on the given dataset."""

    @classmethod
    @abstractmethod
    def get_validation_params(cls, dataset: Any) -> ParamsBundle:
        """Return the parameter bundle for validation on the given dataset."""

    @classmethod
    def get_dataset_classes(cls) -> list:
        """Return the list of dataset classes this configuration supports."""
        return cls.SUPPORTED_DATASETS


__all__ = ["SeanergysModelConfiguration"]
