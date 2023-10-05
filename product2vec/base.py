"""This module provides ABC template for Product2Vec model and custom exceptions.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List, Tuple, Union


class BaseModel(ABC):
    """Base class for Product2Vec model."""

    def __init__(self) -> None:
        """Initialize the class."""
        self._is_fitted = False

    @abstractmethod
    def _reset(self) -> None:
        """Reset learned attributes."""

    @abstractmethod
    def fit(self, baskets: Union[Iterator, Iterable[Iterable[str]]]) -> BaseModel:
        """Fit model to baskets."""

    @abstractmethod
    def show_complements(self, product: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Find top N complements for a specified focal product."""

    @abstractmethod
    def show_substitutes(
        self, product: str, topn: int = 5, penalize: bool = True
    ) -> List[Tuple[str, float]]:
        """Find top N substitutes for a specified focal product."""


class NotFittedError(Exception):
    """Exception thrown in case class isn't fitted."""

    def __init__(self, cls: type):
        msg = f"{cls.__class__.__name__} is not fitted"
        super().__init__(msg)


class OptimizationWarning(UserWarning):
    """Warning thrown in case optimization fails."""

    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

    def __str__(self) -> str:
        return self.msg
