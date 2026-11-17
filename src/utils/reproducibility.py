"""Reproducibility helpers shared by project entry points."""

import random

import numpy as np


def set_random_seeds(seed: int) -> None:
    """Seed Python and NumPy random number generators.

    ``PYTHONHASHSEED`` must be set before the Python process starts to affect
    hash randomization; setting it at runtime would not change the active seed.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")
    random.seed(seed)
    np.random.seed(seed)
