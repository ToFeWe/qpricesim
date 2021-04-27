"""
Tests for the economic environment module.

"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from qpricesim.simulations.utils_simulation import set_numba_seed
from qpricesim.model_code.economic_environment import calc_reward
from qpricesim.model_code.QLearningAgent import QLearningAgent
from qpricesim.model_code.QLearningAgent import QLearningAgentBase


@pytest.fixture
def setup():
    """

    Q learning agent with the matrix

    [[0,1,2,3],
     [4,5,6,7],
     [8,9,10,11],
     [12,13,14,15],
     [16,17,18,19]]
    """
    n_actions = 4
    n_states = 5
    agent = QLearningAgent(
        alpha=0.1,
        epsilon=1,
        discount=0.5,
        n_actions=n_actions,
        n_states=n_states,
    )
    # The q-values are initialized at random for normal training
    # For testing we replace them by something more tractable.
    q_matrix_tractable = np.arange(0, n_states * n_actions).reshape(n_states, n_actions)
    agent.set_qmatrix(q_matrix_tractable.astype("float64"))
    return agent, n_states, n_actions


def test_base_agent_same_as_njit_agent(setup):
    agent, _, _ = setup

    base_agent = QLearningAgentBase(
        alpha=agent.alpha,
        epsilon=agent.epsilon,
        discount=agent.discount,
        n_actions=agent.n_actions,
        n_states=agent.n_states,
    )
    base_agent.set_qmatrix(new_matrix=agent.get_qmatrix())

    np.random.seed(10)
    some_actions_base = [base_agent.get_action(i)[0] for i in range(5)]

    set_numba_seed(10)
    some_actions_njit = [agent.get_action(i)[0] for i in range(5)]
    assert_array_almost_equal(some_actions_njit, some_actions_base)


def test_get_q_value(setup):
    agent, _, _ = setup
    assert agent.get_qvalue(0, 0) == 0
    assert agent.get_qvalue(2, 1) == 9
    assert agent.get_qvalue(3, 3) == 15
    assert agent.get_qvalue(4, 3) == 19


def test_get_value(setup):
    agent, _, _ = setup
    assert agent.get_value(0) == 3
    assert agent.get_value(2) == 11
    assert agent.get_value(3) == 15
    assert agent.get_value(4) == 19


def test_get_q_matrix(setup):
    agent, _, _ = setup
    matrix_to_be = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
        ],
        dtype="float64",
    )
    assert np.array_equal(agent.get_qmatrix(), matrix_to_be)


def test_q_learning_all_best_action(setup):
    agent, n_states, n_actions = setup
    all_best = []

    for int_state in range(n_states):
        all_best.append(agent.get_best_action(int_state))
    # Given the way we reshaped, the last action must
    # always be the best one.
    assert all(val == 3 for val in all_best)


def test_q_learning_update(setup):
    agent, _, _ = setup
    reward = 10
    action = 0
    state = 0
    next_state = 0
    # Perform one update
    agent.update(state, action, reward, next_state)

    # Q(s=0, a=0) = (1 - alpha) * Q +
    #                alpha *  (reward + discount*V(s_t_1=0))
    # Q = 0.9 * 0 + 0.1 * (10 + 0.5 * 3)
    # Q = 1.15
    value_after_first_update = agent.get_qvalue(state, action)
    assert_array_almost_equal(value_after_first_update, 1.15)

    # Perform same update again
    # Q(s=0, a=0) = (1 - alpha) * Q +
    #                alpha *  (reward + discount*V(s_t_1=0))
    # Q = 0.9 * 1.15 + 0.1 * (10 + 0.5 * 3)
    # Q = 2.185

    agent.update(state, action, reward, next_state)
    value_after_second_update = agent.get_qvalue(state, action)
    assert_array_almost_equal(value_after_second_update, 2.185)

    # Perform same update again
    # Q(s=0, a=0) = (1 - alpha) * Q +
    #                alpha *  (reward + discount*V(s_t_1=0))
    # Q = 0.9 * 2.185 + 0.1 * (10 + 0.5 * 3)
    # Q = 3.1165

    agent.update(state, action, reward, next_state)
    value_after_second_update = agent.get_qvalue(state, action)
    assert_array_almost_equal(value_after_second_update, 3.1165)

    # Perform same update again
    # Q(s=0, a=0) = (1 - alpha) * Q +
    #                alpha *  (reward + discount*V(s_t_1=0))
    # Q = 0.9 * 3.1165 + 0.1 * (10 + 0.5 * 3.1165)
    # Q = 3.960675

    agent.update(state, action, reward, next_state)
    value_after_second_update = agent.get_qvalue(state, action)
    assert_array_almost_equal(value_after_second_update, 3.960675)


def test_q_learning_reproducible():
    n_actions = 40
    n_states = 100

    set_numba_seed(10)
    agent_1 = QLearningAgent(
        alpha=0.1, epsilon=1, discount=0.5, n_actions=n_actions, n_states=n_states
    )
    some_actions_1 = [agent_1.get_action(i)[0] for i in range(20)]
    set_numba_seed(10)
    agent_2 = QLearningAgent(
        alpha=0.1, epsilon=1, discount=0.5, n_actions=n_actions, n_states=n_states
    )
    some_actions_2 = [agent_2.get_action(i)[0] for i in range(20)]

    # Same inital q-matrix
    assert_array_almost_equal(agent_1._qvalues, agent_2._qvalues)

    # Same action picked at random
    assert_array_almost_equal(some_actions_1, some_actions_2)


def test_get_action(setup):
    _, n_states, n_actions = setup

    # Epsilon=0 -> non random action
    set_numba_seed(4)
    agent = QLearningAgent(
        alpha=0.1, epsilon=0, discount=0.5, n_actions=n_actions, n_states=n_states
    )
    chosen_action, best_action = agent.get_action(1)
    assert chosen_action == best_action
