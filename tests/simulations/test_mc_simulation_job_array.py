import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from qpricesim.model_code.QLearningAgent import QLearningAgentBase
from qpricesim.simulations.mc_simulation_job_array import run_specific_specification


@pytest.fixture
def setup():
    parameter_1 = {
        "n_agent": 3,
        "k_memory": 1,
        "discount_rate": 0.95,
        "exploration_rate": 1,
        "min_price": 1,
        "max_price": 3,
        "reservation_price": 2,
        "m_consumer": 60,
        "step": 1,
        "learning_iterations": 1000000000,
        "rounds_convergence": 50,
        "Q_star_threshold": 0.000000001,
        "alpha": 0.1,
        "avg_price_rounds": 5,
        "epsilon": 0.05,
    }
    return parameter_1


def test_replication_run_specific_specification(setup):
    paramter = setup
    beta = 0.05
    alpha = 0.05

    # Test if replication works in everyway
    out_1 = run_specific_specification(beta, alpha, paramter, 23)
    out_2 = run_specific_specification(beta, alpha, paramter, 23)

    # Last element should always be QLearning agent
    # Note that it is transformed in the function to the Base
    # class from its njit counter part
    assert all(isinstance(e, QLearningAgentBase) for e in out_1[-1])
    assert all(isinstance(e, QLearningAgentBase) for e in out_2[-1])

    for i, (v_1, v_2) in enumerate(zip(out_1[:-1], out_2[:-1])):
        assert_array_almost_equal(v_1, v_2)

    q_matrix_1 = [a._qvalues for a in out_1[-1]]
    q_matrix_2 = [a._qvalues for a in out_1[-1]]
    assert_array_almost_equal(q_matrix_1, q_matrix_2)
