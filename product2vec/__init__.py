"""This module provides Product2Vec model and tools.

"""
from product2vec.base import BaseModel, NotFittedError, OptimizationWarning
from product2vec.model import Product2Vec
from product2vec.model_tools import EpochLogger
from product2vec.data import BasketGenerator

__all__ = [
    "BaseModel",
    "NotFittedError",
    "OptimizationWarning",
    "EpochLogger",
    "BasketGenerator",
    "Product2Vec",
]

__version__ = "0.0.1"
