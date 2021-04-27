import numpy as np
import pytest
from numpy.testing import assert_allclose

from qpricesim.simulations.utils_performance import best_response_share
from qpricesim.simulations.utils_performance import check_if_nash_equilibrium
from qpricesim.simulations.utils_performance import get_average_profitability
from qpricesim.simulations.utils_performance import get_delta
from qpricesim.simulations.utils_performance import get_specific_profitability
from qpricesim.simulations.utils_performance import get_weighted_profitability
from qpricesim.simulations.utils_performance import sort_list


@pytest.fixture
def setup():
    array_1 = np.array([[4, 9, 2], [3, 4, 1], [1, 1, 1]])
    array_2 = np.array([[4, 5, 5], [1, 0, 1], [4, 1, 8]])
    array_3 = np.array([[5, 5, 5], [5, 0, 0], [4, 2, 8]])

    return array_1, array_2, array_3


def test_nash_equilibrium(setup):
    array_1, array_2, array_3 = setup
    # array actual id, array optimal id, state
    # All done by pen n paper
    out_1_2_0 = check_if_nash_equilibrium(array_1, array_2, 0)
    assert out_1_2_0, "Should be a NE"

    out_1_3_0 = check_if_nash_equilibrium(array_1, array_3, 0)
    assert not out_1_3_0, "Should not be a NE"

    out_3_2_2 = check_if_nash_equilibrium(array_3, array_2, 2)
    assert out_3_2_2, "Should be a NE"

    out_1_3_1 = check_if_nash_equilibrium(array_1, array_3, 1)
    assert not out_1_3_1, "Should not be a NE"


def test_get_specific_profitability(setup):
    array_1, array_2, array_3 = setup
    profit_1_1 = get_specific_profitability(actual_agent_q_matrix=array_1, state=1)
    profit_2_2 = get_specific_profitability(actual_agent_q_matrix=array_2, state=2)

    profit_3_0 = get_specific_profitability(actual_agent_q_matrix=array_3, state=0)

    assert profit_1_1 == 4
    assert profit_2_2 == 8
    assert profit_3_0 == 5


def test_best_response_share(setup):
    array_1, array_2, array_3 = setup
    br_share_1_2 = best_response_share(
        optimal_agent_q_matrix=array_1, actual_agent_q_matrix=array_2
    )
    br_share_2_3 = best_response_share(
        optimal_agent_q_matrix=array_2, actual_agent_q_matrix=array_3
    )
    br_share_3_1 = best_response_share(
        optimal_agent_q_matrix=array_3, actual_agent_q_matrix=array_1
    )
    br_share_1_3 = best_response_share(
        optimal_agent_q_matrix=array_1, actual_agent_q_matrix=array_3
    )

    assert br_share_1_2 == 1 / 3
    assert br_share_2_3 == 2 / 3
    assert br_share_3_1 == 0
    assert br_share_3_1 == br_share_1_3


def test_get_delta(setup):
    array_1, array_2, array_3 = setup
    delta_1_2 = get_delta(old_q_value=array_1, new_q_value=array_2)
    delta_2_3 = get_delta(old_q_value=array_2, new_q_value=array_3)
    delta_3_1 = get_delta(old_q_value=array_3, new_q_value=array_1)
    delta_1_3 = get_delta(old_q_value=array_1, new_q_value=array_3)

    assert delta_1_2 == 7
    assert delta_3_1 == 7
    assert delta_2_3 == 4
    assert delta_1_3 == delta_3_1


def test_sort_list():
    list_1 = [1, 2, 0]
    list_2 = ["yolo", "lol", "rofl"]

    sorted_list = sort_list(list_to_sort=list_2, sort_by_list=list_1)
    assert sorted_list == ["rofl", "yolo", "lol"]


def test_sort_list_2():
    list_1 = [0, 1, 2, 3, 4]
    list_2 = ["yolo", "lol", "rofl", "last"]

    sorted_list = sort_list(list_to_sort=list_2, sort_by_list=list_1)
    assert sorted_list == ["yolo", "lol", "rofl", "last"]


def test_sort_list_3():
    list_1 = [3, 2, 1, 4, 0]
    list_2 = [2, 2, 4, 6, 1]

    sorted_list = sort_list(list_to_sort=list_2, sort_by_list=list_1)
    assert sorted_list == [1, 4, 2, 2, 6]


def test_get_weighted_profitability(setup):
    array_1, array_2, array_3 = setup

    wp_1_1 = get_weighted_profitability(
        optimal_agent_q_matrix=array_1, actual_agent_q_matrix=array_1
    )
    wp_2_2 = get_weighted_profitability(
        optimal_agent_q_matrix=array_2, actual_agent_q_matrix=array_2
    )
    wp_3_3 = get_weighted_profitability(
        optimal_agent_q_matrix=array_3, actual_agent_q_matrix=array_3
    )

    wp_3_1 = get_weighted_profitability(
        optimal_agent_q_matrix=array_3, actual_agent_q_matrix=array_1
    )

    wp_1_2 = get_weighted_profitability(
        optimal_agent_q_matrix=array_1, actual_agent_q_matrix=array_2
    )

    assert_allclose(wp_1_1, 4 + 2 / 3)
    assert_allclose(wp_2_2, 4 + 2 / 3)
    assert_allclose(wp_3_3, 6)
    assert_allclose(wp_3_1, 0 * (4 + 2 / 3))
    assert_allclose(wp_1_2, 1 / 3 * (4 + 2 / 3))


def test_get_average_profitability(setup):
    array_1, array_2, array_3 = setup

    avg_profit_1 = get_average_profitability(array_1)
    avg_profit_2 = get_average_profitability(array_2)
    avg_profit_3 = get_average_profitability(array_3)

    assert_allclose(avg_profit_1, 4 + 2 / 3)
    assert_allclose(avg_profit_2, 4 + 2 / 3)
    assert_allclose(avg_profit_3, 6)

    array_1 = np.array([[4, 9, 2], [3, 4, 1], [1, 1, 1]])
    array_2 = np.array([[4, 5, 5], [1, 0, 1], [4, 1, 8]])
    array_3 = np.array([[5, 5, 5], [5, 0, 0], [4, 2, 8]])
