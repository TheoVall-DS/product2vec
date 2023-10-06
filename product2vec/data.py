"""This module provides basket generation tools.

"""
from typing import List, Union

import numpy as np
from joblib import Parallel, delayed


class BasketGenerator:
    """BasketGenerator creates synthetic baskets.
    First it generates copurchase (cooccurence) matrix with probabilities
    of products being in the same basket and then uses it to sample products.
    It helps create patterns similar to real world data.

    Parameters
    ----------
    n_jobs : int, default=None
        Number of parallel jobs.
    verbose : int, default=1
        Verbosity level.
    seed : int, default=1
        Random seed.
    extreme : int or float, default=10
        Larger values result in more extreme probability range for
        product cooccurence in the same basket.
    """

    def __init__(
        self,
        n_jobs: int = None,
        verbose: int = 1,
        seed: int = 1,
        extreme: Union[int, float] = 10,
    ) -> None:
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.seed = seed
        self.extreme = extreme

        if extreme <= 1:
            raise ValueError("'extreme' should be > 1")

    def _generate_basket(
        self, copurchase_probs: np.ndarray, min_size: int, max_size: int, seed: int
    ) -> List[str]:
        """Generate one basket."""
        basket = set()
        available_products = np.arange(copurchase_probs.shape[0])

        rng = np.random.default_rng(seed)
        basket_size = rng.integers(low=min_size, high=max_size + 1)
        current_product = rng.integers(low=0, high=copurchase_probs.shape[0])
        basket.add(str(current_product))

        while len(basket) < basket_size:
            # new rng to ensure different results
            seed = seed + 1 if seed else None
            rng = np.random.default_rng(seed)

            # pick new product based on its probability to occur with previous one
            current_product = rng.choice(
                available_products, p=copurchase_probs[current_product, :]
            )
            if current_product not in basket:
                basket.add(str(current_product))  # add only unique products

        return list(basket)

    def __call__(
        self,
        n_baskets: int = 1000,
        n_products: int = 100,
        min_size: int = 2,
        max_size: int = 10,
    ) -> List[List[str]]:
        """Generate synthetic baskets with specified parameters.

        Parameters
        ----------
        n_baskets : int, default=1000
            Number of baskets.
        n_products : int, default=100
            Number of unique products.
        min_size : int, default=2
            Minimum number (inclusive) of unique products in the same basket.
        max_size : int, default=10
            Maximum number (inclusive) of unique products in the same basket.

        Returns
        -------
        baskets : list of list of str
            Synthetic baskets with product labels from '0' to 'n_products' - 1.
            Each basket has at least 2 products.

        Notes
        -----
        Baskets are guaranteed to have only unique products.
        """
        if min_size < 2:
            raise ValueError("Minimum basket size is 2")

        # generate copurchase matrix for products
        rng = np.random.default_rng(self.seed)
        copurchase_matrix = rng.integers(
            low=1, high=n_products * self.extreme, size=(n_products, n_products)
        )  # more products result in more extreme probabilities
        copurchase_probs = np.apply_along_axis(
            lambda x: x / np.sum(x), axis=1, arr=copurchase_matrix
        )  # normalize values to make probabilities

        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            baskets = parallel(
                delayed(self._generate_basket)(
                    copurchase_probs=copurchase_probs,
                    min_size=min_size,
                    max_size=max_size,
                    seed=self.seed + i if self.seed else None,
                )
                for i in range(n_baskets)
            )

        return baskets
