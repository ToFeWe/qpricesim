"""

A collection of different help function for training the Q-learning
agents.
"""
import numpy as np
from numba import njit


@njit
def numba_argmax(array):
    """
    Calculates the argmax of an array with numba support.

    """
    return np.argmax(array)


@njit
def numba_max(array):
    """
    Calculates the max of an array with numba support.

    """
    return np.max(array)
