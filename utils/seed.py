"""
Random Seed and Reproducibility Utilities.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for Python random, NumPy, and TensorFlow for reproducible runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
