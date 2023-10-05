"""This module provides Product2Vec model.

"""
from __future__ import annotations

import warnings
from functools import partial
from typing import Iterable, Iterator, List, Tuple, Union

import numpy as np
from gensim.models import Word2Vec
from scipy.optimize import minimize

from product2vec.base import BaseModel, NotFittedError, OptimizationWarning


class Product2Vec(BaseModel):
    """Product2Vec model is capable of finding complements and substitutes
    among products.

    Parameters
    ----------
    **gensim_kwargs : dict
        Keyword arguments for underlying gensim model.
        The following parameters are not available:
        - sentences (passed to `fit` method)
        - window (set to 1000)
        - sg (set to 1)
        - hs (set to 0)
        - shrink_windows (set to False)

    Attributes
    ----------
    model_ : Word2Vec
        Fitted gensim model with all available methods and attributes.

    Raises
    ------
    NotFittedError
        If `show_complements` or `show_substitutes` methods are used
        before calling `fit` method first.

    Warns
    -----
    OptimizationWarning
        If correlation minimization fails.

    References
    ----------
    Original paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3519358
    """

    def __init__(self, **gensim_kwargs: dict) -> None:
        super().__init__()  # set fit status to False
        self.gensim_kwargs = gensim_kwargs

        self.model_ = None  # to be fitted

    def _reset(self) -> None:
        """Reset gensim model and fit status."""
        super().__init__()  # reset fit status
        self.model_ = None

    def _compute_cond_probs(self) -> np.ndarray:
        """Compute conditional probabilities.

        Notes
        -----
        Prob(product_0 | product_1) can be found at position (row 0, column 1).
        """

        def sigmoid(value: float) -> float:
            """Apply sigmoid function to value."""
            return 1 / (1 + np.exp(-value))

        cond_probs = sigmoid(self.model_.syn1neg @ self.model_.wv.vectors.T)

        return cond_probs

    def _compute_complementarity(self, product: str) -> np.ndarray:
        """Compute complementarity scores.

        Notes
        -----
        Complementarity matrix is symmetric so you can pick either row or column
        for a focal product.
        """
        cond_probs = self._compute_cond_probs()
        complementarity_table = 1 / 2 * (cond_probs + cond_probs.T)

        product_idx = self.model_.wv.key_to_index[product]
        scores = complementarity_table[product_idx, :]  # row for focal product
        scores[product_idx] = 0  # product isn't complement to itself

        return scores

    def _compute_exchangeability(self, product: str):
        """Compute exchangeability scores.

        Notes
        -----
        Probabilities of products given specified focal one are fixed.
        Then they are compared with conditional probabilities given other products.
        """
        cond_probs = self._compute_cond_probs()
        product_idx = self.model_.wv.key_to_index[product]
        probs_diff = cond_probs[:, product_idx].reshape(-1, 1) - cond_probs
        scores = -np.linalg.norm(probs_diff, ord=2, axis=0)
        scores[product_idx] = np.NINF  # product isn't substitute for itself

        return scores

    @staticmethod
    def _abs_corr(
        lambda_param: float, ex_scores: np.ndarray, c_scores: np.ndarray
    ) -> float:
        """Compute absolute Pearson's correlation coefficient."""
        pe_ab = ex_scores - lambda_param * c_scores  # penalized exchangeability
        pe_ab_se = np.sqrt(np.sum(np.square(pe_ab - np.mean(pe_ab))))
        c_ab_se = np.sqrt(np.sum(np.square(c_scores - np.mean(c_scores))))

        numerator = np.sum(
            (pe_ab - np.mean(pe_ab)) * (c_scores - np.mean(c_scores))
        )  # covariance
        denominator = pe_ab_se * c_ab_se  # product of standard deviations

        return np.abs(numerator / denominator)

    def _optimize_lambda(
        self, c_scores: np.ndarray, ex_scores: np.ndarray, guess: int | float
    ) -> float:
        """Optimize lambda parameter to reduce correlation between
        complementarity and exchangeability.
        """
        # exclude focal product scores
        ex_scores = ex_scores[ex_scores != np.NINF]
        c_scores = c_scores[c_scores != 0]
        opt_result = minimize(
            partial(self._abs_corr, ex_scores=ex_scores, c_scores=c_scores), x0=guess
        )

        if np.abs(opt_result.fun) > np.abs(np.corrcoef(c_scores, ex_scores)[0, 1]):
            warnings.warn(message=OptimizationWarning("Failed to minimze correlation"))
            lambda_param = 0
        else:
            lambda_param = opt_result.x[0]

        return lambda_param

    def _find_topn(self, scores: np.ndarray, topn: int) -> List[Tuple[str, float]]:
        """Find topn N complements/substitutes."""
        best_candidates = np.argsort(scores)[::-1][:topn]
        labels = np.array(self.model_.wv.index_to_key)
        candidates = list(zip(labels[best_candidates], scores[best_candidates]))

        return candidates

    def fit(self, baskets: Union[Iterator, Iterable[Iterable[str]]]) -> Product2Vec:
        """Fit Product2Vec model.

        Parameters
        ----------
        baskets : iterable of iterable or iterator
            Array of baskets with products.

        Returns
        -------
        self : object
            Fitted Product2Vec model.
        """
        self._reset()

        self.model_ = Word2Vec(
            sentences=baskets,
            window=1000,  # take the whole basket, order doesn't matter
            sg=1,  # use skip-gram model
            hs=0,  # allow negative sampling
            shrink_windows=False,  # fixed window size
            **self.gensim_kwargs,
        )

        setattr(self, "_is_fitted", True)

        return self

    def show_complements(self, product: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Show top N complements for a specified focal product.

        Parameters
        ----------
        product : str
            Focal product.
        topn : int, default=5
            Number of complements.

        Returns
        -------
        complements : list of tuples
            Complementarity products and their complementarity scores.
        """
        if not getattr(self, "_is_fitted"):
            raise NotFittedError(self)

        scores = self._compute_complementarity(product)
        complements = self._find_topn(scores, topn)

        return complements

    def show_substitutes(
        self,
        product: str,
        topn: int = 5,
        penalize: bool = True,
        guess: int | float = 0,
    ) -> List[Tuple[str, float]]:
        """Show top N substitutes for a specified focal product.

        Parameters
        ----------
        product : str
            Focal product.
        topn : int, default=5
            Number of substitutes.
        penalize : bool, default=True
            Whether to penalize exchangeability scores. Strongly recommended.
        guess : int or float, default=0
            Initial guess for regularization parameter lambda.
            Tweak only if you encounter OptimizationWarning.
            In practive value in range (-10, 10) is a good guess.

        Returns
        -------
        substitutes : list of tuples
            Interchangeable products and their exchangeability scores.
        """
        if not getattr(self, "_is_fitted"):
            raise NotFittedError(self)

        ex_scores = self._compute_exchangeability(product)
        if penalize:
            c_scores = self._compute_complementarity(product)
            lambda_param = self._optimize_lambda(
                c_scores=c_scores, ex_scores=ex_scores, guess=guess
            )
            ex_scores = ex_scores - lambda_param * c_scores

        substitutes = self._find_topn(ex_scores, topn)

        return substitutes
