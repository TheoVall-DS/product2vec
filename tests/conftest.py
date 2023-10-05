"""This module provides fixtures for tests.

"""
# pylint: disable=redefined-outer-name
from typing import List

import pytest

from product2vec import BasketGenerator, EpochLogger, Product2Vec


@pytest.fixture(scope="session")
def baskets() -> List[List[str]]:
    "Generate synthetic baskets with default BasketGenerator parameters."
    generator = BasketGenerator()
    data = generator()

    return data


@pytest.fixture(scope="session")
def fitted_model(baskets):
    """Fit Product2Vec model."""
    logger = EpochLogger(n_latest=3)
    model = Product2Vec(
        vector_size=10,
        min_count=1,
        epochs=20,
        callbacks=[logger],
        seed=1,
        workers=1,
    )
    _ = model.fit(baskets)

    return model


@pytest.fixture(scope="session")
def focal_product(fitted_model):
    """Take first product and use it as focal one."""
    return fitted_model.model_.wv.index_to_key[0]


@pytest.fixture(scope="session")
def total_products(fitted_model):
    """Show total number of products."""
    return len(fitted_model.model_.wv.index_to_key)


@pytest.fixture(scope="session")
def show_all_candidates(fitted_model, focal_product, total_products):
    """Show all complements and substitutes for focal product."""
    complements = fitted_model.show_complements(
        product=focal_product, topn=total_products
    )
    substitutes = fitted_model.show_substitutes(
        product=focal_product, topn=total_products
    )

    return complements, substitutes
