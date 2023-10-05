"""This module provides tests for Product2Vec model.

"""
import numpy as np
import pytest

from product2vec import NotFittedError, Product2Vec


class TestModel:
    """Test Product2Vec model."""

    def test_logger(self, fitted_model):
        """Test epoch logger."""
        # pylint: disable=protected-access
        logger = fitted_model.gensim_kwargs["callbacks"][0]
        assertion_a = logger._current_epoch - 1 == fitted_model.gensim_kwargs["epochs"]
        assertion_b = logger._epoch_duration is not None

        assert assertion_a, "Epoch counter failed to capture number of epochs"
        assert assertion_b, "Epoch logger failed to capture durations"

        expected_time = [(3600, "01:00:00"), (3900, "01:05:00"), (400, "06:40")]
        for time in expected_time:
            assertion_c = logger.format_time(time[0] == time[1])
            assert assertion_c, "Time format is incorrect"

    def test_labels(self, fitted_model, baskets):
        """Test product labels in data vs fitted model."""
        true_labels = {product for basket in baskets for product in basket}
        model_labels = set(fitted_model.model_.wv.key_to_index.keys())

        assert true_labels == model_labels, "Product labels differ in model and data"

    def test_complementarity_scores(self, show_all_candidates):
        """Test complementarity values range."""
        complements, _ = show_all_candidates
        _, scores = list(zip(*complements))

        assert np.all(scores) >= 0, "Found complementarity scores < 0"
        assert np.all(scores) < 1, "Found complementarity scores >= 1"

    def test_descending_scores(self, show_all_candidates):
        """Test descending order of complements/substitutes."""
        complements, substitutes = show_all_candidates
        _, c_scores = list(zip(*complements))
        _, ex_scores = list(zip(*substitutes))

        assertion_a = np.all(c_scores[1:] <= c_scores[:-1])
        assertion_b = np.all(ex_scores[1:] <= ex_scores[:-1])

        assert assertion_a, "Complement scores are not descending"
        assert assertion_b, "Substitute scores are not descending"

    def test_topn(self, fitted_model, focal_product):
        """Test limited number of output products."""
        complements = fitted_model.show_complements(product=focal_product, topn=5)
        substitutes = fitted_model.show_substitutes(product=focal_product, topn=5)

        assert len(complements) == 5, "Number of complements differs from expected"
        assert len(substitutes) == 5, "Number of substitutes differs from expected"

    def test_self_values(self, fitted_model):
        """Make sure focal product isn't complement/substitute fot itself."""
        all_products = fitted_model.model_.wv.index_to_key
        for product in all_products:
            complements = fitted_model.show_complements(product=product, topn=1000)
            substitutes = fitted_model.show_substitutes(product=product, topn=1000)

            assert complements[-1][1] == 0, "Found self-complement"
            assert substitutes[-1][1] == np.NINF, "Found self-substitute"

    def test_fit_error(self, focal_product):
        """Test NotFittedError."""
        model = Product2Vec()

        with pytest.raises(NotFittedError):
            model.show_complements(product=focal_product)

        with pytest.raises(NotFittedError):
            model.show_substitutes(product=focal_product)
