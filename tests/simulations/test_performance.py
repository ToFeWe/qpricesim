"""

Integration tests for performance module.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from qpricesim.model_code.QLearningAgent import QLearningAgent
from qpricesim.simulations.performance import get_all_prices
from qpricesim.simulations.performance import get_optimal_q_matrix
from qpricesim.simulations.performance import get_performance_measures
from qpricesim.simulations.performance import get_transition_arrays
from qpricesim.simulations.utils_performance import best_response_share
from qpricesim.simulations.utils_performance import check_if_nash_equilibrium
from qpricesim.simulations.utils_performance import get_average_profitability
from qpricesim.simulations.utils_performance import get_specific_profitability
from qpricesim.simulations.utils_performance import get_weighted_profitability
from qpricesim.simulations.utils_simulation import gen_possible_prices
from qpricesim.simulations.utils_simulation import gen_price_combination_byte_mappings


@pytest.fixture
def setup():
    array_1 = np.array([[4, 2], [3, 1], [1, 1], [1, 2]]).astype("float64")
    array_2 = np.array([[4, 5], [1, 1], [4, 8], [3, 4]]).astype("float64")

    parameter_1 = {
        "n_agent": 2,
        "k_memory": 1,
        "discount_rate": 0.95,
        "exploration_rate": 1,
        "min_price": 1,
        "max_price": 2,
        "reservation_price": 2,
        "m_consumer": 60,
        "step": 1,
        "learning_iterations": 1000000000,
        "rounds_convergence": 100000,
        "Q_star_threshold": 0.000000001,
        "alpha": 0.1,
        "avg_price_rounds": 5,
        "epsilon": 0.1,
    }

    agent_1 = QLearningAgent(
        alpha=parameter_1["alpha"],
        epsilon=parameter_1["epsilon"],
        discount=parameter_1["discount_rate"],
        n_actions=array_1.shape[0],
        n_states=array_1.shape[1],
    )

    agent_2 = QLearningAgent(
        alpha=parameter_1["alpha"],
        epsilon=parameter_1["epsilon"],
        discount=parameter_1["discount_rate"],
        n_actions=array_1.shape[0],
        n_states=array_1.shape[1],
    )

    # Use the arrays as q-values
    agent_1._qvalues = array_1
    agent_2._qvalues = array_2

    all_agents = [agent_1, agent_2]

    return all_agents, parameter_1


def test_get_all_prices(setup):
    all_agents, parameter = setup
    prices_to_int_dict, int_to_prices_dict = gen_price_combination_byte_mappings(
        parameter
    )
    price_set = gen_possible_prices(parameter)
    state_1 = 2
    price_array_1 = get_all_prices(
        all_agents=all_agents,
        parameter=parameter,
        prices_to_int_dict=prices_to_int_dict,
        int_to_prices_dict=int_to_prices_dict,
        price_set=price_set,
        state_of_convergence=state_1,
    )
    # derived by pen and paper
    expected_array_1 = np.array([[1, 2], [1, 1], [1, 2], [1, 1], [1, 2]])
    assert_array_almost_equal(expected_array_1, price_array_1)

    state_2 = 3
    price_array_2 = get_all_prices(
        all_agents=all_agents,
        parameter=parameter,
        prices_to_int_dict=prices_to_int_dict,
        int_to_prices_dict=int_to_prices_dict,
        price_set=price_set,
        state_of_convergence=state_2,
    )
    # derived by pen and paper
    expected_array_2 = np.array([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])
    assert_array_almost_equal(expected_array_2, price_array_2)


def test_get_transition_array(setup):
    all_agents, parameter = setup
    prices_to_int_dict, int_to_prices_dict = gen_price_combination_byte_mappings(
        parameter
    )
    price_set = gen_possible_prices(parameter)

    # Derived with pen and paper
    expected_next_state_arrays = [
        np.array([[1, 3], [0, 2], [1, 3], [1, 3]]),
        np.array([[0, 1], [0, 1], [0, 1], [2, 3]]),
    ]

    expected_profit_arrays = [
        np.array([[60, 60], [30, 0], [60, 60], [60, 60]]),
        np.array([[30, 0], [30, 0], [30, 0], [60, 60]]),
    ]

    all_positions = list(range(len(all_agents)))
    for pos, agent in enumerate(all_agents):
        all_competitors = list(all_agents)
        all_competitors.remove(agent)
        position_list_competitors = list(all_positions)
        position_list_competitors.remove(pos)

        profit_array, next_state_array = get_transition_arrays(
            all_trained_competitors=all_competitors,
            parameter=parameter,
            position_agent=pos,
            position_list_competitors=position_list_competitors,
            prices_to_int_dict=prices_to_int_dict,
            int_to_prices_dict=int_to_prices_dict,
            price_set=price_set,
        )
        assert_array_almost_equal(next_state_array, expected_next_state_arrays[pos])
        assert_array_almost_equal(profit_array, expected_profit_arrays[pos])


def test_get_optimal_q_matrix(setup):
    # Note that this is not easily
    # testable...
    # We only look at a boarder case here.
    all_agents, parameter = setup
    prices_to_int_dict, int_to_prices_dict = gen_price_combination_byte_mappings(
        parameter
    )
    price_set = gen_possible_prices(parameter)

    all_positions = list(range(len(all_agents)))
    for pos, agent in enumerate(all_agents):
        all_competitors = list(all_agents)
        all_competitors.remove(agent)
        position_list_competitors = list(all_positions)
        position_list_competitors.remove(pos)

        profit_array, next_state_array = get_transition_arrays(
            all_trained_competitors=all_competitors,
            parameter=parameter,
            position_agent=pos,
            position_list_competitors=position_list_competitors,
            prices_to_int_dict=prices_to_int_dict,
            int_to_prices_dict=int_to_prices_dict,
            price_set=price_set,
        )
        optimal_q_matrix = get_optimal_q_matrix(
            profit_array, next_state_array, parameter, int_to_prices_dict, price_set
        )
        # In index state 3 where both agents play index action
        # 1, we have a steady state as we end up again in
        # state 3. Hence:
        assert_allclose(
            optimal_q_matrix[3, 1],
            1 / (1 - parameter["discount_rate"]) * profit_array[3, 1],
        )


def test_get_performance_measures_deterministic(setup):
    all_agents, parameter = setup
    all_performance_measures_all_agents_1 = get_performance_measures(
        all_agents, parameter, 1
    )

    all_performance_measures_all_agents_2 = get_performance_measures(
        all_agents, parameter, 1
    )

    for a_1, a_2 in zip(
        all_performance_measures_all_agents_1, all_performance_measures_all_agents_2
    ):
        assert np.array_equal(a_1, a_2)


def test_integration_performance_measures(setup):
    all_agents, parameter = setup

    # Performance measures
    out_state_1 = get_performance_measures(
        all_agents=all_agents, parameter=parameter, convergence_state=1
    )

    all_positions = list(range(len(all_agents)))
    prices_to_int_dict, int_to_prices_dict = gen_price_combination_byte_mappings(
        parameter
    )
    price_set = gen_possible_prices(parameter)

    expected_prices_state_1 = get_all_prices(
        all_agents,
        parameter,
        prices_to_int_dict,
        int_to_prices_dict,
        price_set,
        1,
    )

    for pos, agent in enumerate(all_agents):
        all_competitors = list(all_agents)
        all_competitors.remove(agent)

        position_list_competitors = list(all_positions)
        position_list_competitors.remove(pos)
        # Given we know the MDP here, we know all state transitions and rewards that result from
        # each state/action combination.
        profit_array, next_state_array = get_transition_arrays(
            all_trained_competitors=all_competitors,
            parameter=parameter,
            position_agent=pos,
            position_list_competitors=position_list_competitors,
            prices_to_int_dict=prices_to_int_dict,
            int_to_prices_dict=int_to_prices_dict,
            price_set=price_set,
        )
        # Derive the optimal q-matrix for the now deterministic problem.
        optimal_agent_q_matrix = get_optimal_q_matrix(
            profit_array=profit_array,
            next_state_array=next_state_array,
            parameter=parameter,
            int_to_prices_dict=int_to_prices_dict,
            price_set=price_set,
        )
        ne_expected = check_if_nash_equilibrium(
            optimal_agent_q_matrix, agent._qvalues, 1
        )
        br_expected = best_response_share(optimal_agent_q_matrix, agent._qvalues)
        specific_prof_expected = get_specific_profitability(agent._qvalues, 1)
        wp_expected = get_weighted_profitability(optimal_agent_q_matrix, agent._qvalues)
        avg_profit = get_average_profitability(agent._qvalues)
        assert out_state_1[0][pos] == specific_prof_expected
        assert out_state_1[1][pos] == wp_expected
        assert out_state_1[2][pos] == br_expected
        assert out_state_1[3][pos] == avg_profit
        assert out_state_1[5][pos] == ne_expected
    assert_array_almost_equal(out_state_1[4], expected_prices_state_1)
