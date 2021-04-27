import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from qpricesim.model_code.utils_q_learning import numba_argmax
from qpricesim.model_code.utils_q_learning import numba_max


@pytest.fixture
def setup():
    n_actions = 4
    n_states = 5

    array_zeros = np.zeros((4, 4))
    array_identity = np.identity(5)
    array_order = np.arange(0, n_actions * n_states).reshape(n_states, n_actions)
    return array_zeros, array_identity, array_order


def test_numba_arg_max_full(setup):
    array_zeros, array_identity, array_order = setup
    assert numba_argmax(array_zeros) == 0
    assert numba_argmax(array_identity) == 0  # first instance with max 1
    assert numba_argmax(array_order) == 19


def test_numba_arg_max_slice(setup):
    array_zeros, array_identity, array_order = setup
    assert numba_argmax(array_zeros[1, :]) == 0
    assert numba_argmax(array_identity[1, :]) == 1
    assert numba_argmax(array_order[1, :]) == 3

    assert numba_argmax(array_zeros[:, 2]) == 0
    assert numba_argmax(array_identity[:, 2]) == 2
    assert numba_argmax(array_order[:, 3]) == 4


def test_numba_max_full(setup):
    array_zeros, array_identity, array_order = setup
    assert numba_max(array_zeros) == 0
    assert numba_max(array_identity) == 1
    assert numba_max(array_order) == 19


def test_numba_max_slice(setup):
    array_zeros, array_identity, array_order = setup
    assert numba_max(array_zeros[1, :]) == 0
    assert numba_max(array_identity[1, :]) == 1
    assert numba_max(array_order[1, :]) == 7

    assert numba_max(array_zeros[:, 2]) == 0
    assert numba_max(array_identity[:, 2]) == 1
    assert numba_max(array_order[:, 2]) == 18
