"""This module provides tests for BasketGenerator class.

"""
import pytest

from product2vec.data import BasketGenerator


class TestGenerator:
    """Test BasketGenerator."""

    def test_n_baskets(self):
        """Test correct number of generated baskets."""
        generator = BasketGenerator()
        data = generator(n_baskets=100)

        assert len(data) == 100, "True number of baskets is different expected 100"

    def test_n_products(self):
        """Test correct number of unique products in baskets."""
        generator = BasketGenerator()
        data = generator(n_baskets=1000, n_products=99)
        unique_products = {product for basket in data for product in basket}

        assert (
            len(unique_products) == 99
        ), "True number of unique products is different from 99"

    def test_basket_size(self):
        """Test correct number of basket size (min and max number of products)."""
        generator = BasketGenerator()
        data = generator(n_baskets=1000, n_products=99, min_size=2, max_size=20)
        true_len = [len(basket) for basket in data]
        true_min, true_max = min(true_len), max(true_len)

        assert true_min == 2, "Minimum basket size is different from 2"
        assert true_max == 20, "Maximum basket size is different from 20"

    def test_exceptions(self):
        """Test class exceptions."""
        with pytest.raises(ValueError):
            generator = BasketGenerator(extreme=0)

        with pytest.raises(ValueError):
            generator = BasketGenerator()
            _ = generator(min_size=1)
