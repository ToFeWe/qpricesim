import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from qpricesim.simulations.utils_simulation import _flatten_array_to_str
from qpricesim.simulations.utils_simulation import best_action_same
from qpricesim.simulations.utils_simulation import calc_epsilon
from qpricesim.simulations.utils_simulation import concatenate_new_price_state
from qpricesim.simulations.utils_simulation import gen_possible_price_combination
from qpricesim.simulations.utils_simulation import gen_possible_prices
from qpricesim.simulations.utils_simulation import gen_price_combination_byte_mappings
from qpricesim.simulations.utils_simulation import int_state_to_price_state
from qpricesim.simulations.utils_simulation import price_state_to_int_state


@pytest.fixture
def parameter_setup():
    parameter_1 = {
        "min_price": 1,
        "max_price": 2,
        "step": 1,
        "n_agent": 2,
        "k_memory": 2,
    }

    parameter_2 = {
        "min_price": 0,
        "max_price": 4,
        "step": 2,
        "n_agent": 3,
        "k_memory": 1,
    }

    return parameter_1, parameter_2


def test_calc_epsilon():
    eps_1 = calc_epsilon(beta=1, t=4)
    eps_2 = calc_epsilon(beta=0.002, t=100)
    eps_3 = calc_epsilon(beta=5e-6, t=1000000)

    assert_allclose(eps_1, 0.01831563888)
    assert_allclose(eps_2, 0.81873075307)
    assert_allclose(eps_3, 0.00673794699)


def test_gen_possible_prices(parameter_setup):
    parameter_1, parameter_2 = parameter_setup

    pp_1 = gen_possible_prices(parameter_1)
    pp_2 = gen_possible_prices(parameter_2)

    assert_array_almost_equal(pp_1, np.array([1, 2]))
    assert_array_almost_equal(pp_2, np.array([0, 2, 4]))


def test_gen_possible_price_combination(parameter_setup):
    parameter_1, parameter_2 = parameter_setup
    ppc_1 = gen_possible_price_combination(parameter_1)

    assert len(ppc_1) == 2 ** 4

    assert (1, 1, 1, 1) in ppc_1
    assert (1, 2, 2, 1) in ppc_1
    assert (1, 1, 1, 0) not in ppc_1
    assert (1, 1, 1, 3) not in ppc_1

    ppc_2 = gen_possible_price_combination(parameter_2)
    assert len(ppc_2) == 3 ** 3
    assert (0, 0, 0) in ppc_2
    assert (2, 4, 0) in ppc_2
    assert (0, 3, 4) not in ppc_2
    assert (1, 2, 2) not in ppc_2


def test_best_action_same():
    array_1 = np.array([[8, 2, 1], [3, 4, 5]])
    array_2 = np.array([[3, 4, 5], [8, 2, 1]])
    array_3 = np.array([[8, 2, 1], [3, 4, 4]])

    assert not best_action_same(array_1, array_2)
    assert not best_action_same(array_1, array_3)
    assert not best_action_same(array_2, array_3)
    assert best_action_same(array_2, array_2)
    assert best_action_same(array_3, array_3)


def test_concatenate_new_price_state():
    array_1 = np.array([3, 4, 5])
    array_2 = np.array([9, 2, 1])
    array_3 = np.array([1, 2, 3, 4, 5, 6])

    new_array_3_1_3 = concatenate_new_price_state(
        old_price_state=array_3, new_prices=array_1, n_agent=3
    )
    new_array_3_1_3_again = concatenate_new_price_state(
        old_price_state=new_array_3_1_3, new_prices=array_1, n_agent=3
    )
    new_array_1_2_3 = concatenate_new_price_state(
        old_price_state=array_1, new_prices=array_2, n_agent=3
    )
    new_array_2_3_2 = concatenate_new_price_state(
        old_price_state=array_2, new_prices=array_3, n_agent=2
    )
    assert_array_almost_equal(new_array_2_3_2, np.array([1, 1, 2, 3, 4, 5, 6]))
    assert_array_almost_equal(new_array_1_2_3, np.array([9, 2, 1]))
    assert_array_almost_equal(new_array_3_1_3, np.array([4, 5, 6, 3, 4, 5]))
    assert_array_almost_equal(new_array_3_1_3_again, np.array([3, 4, 5, 3, 4, 5]))


def test_gen_price_combination_byte_mappings_same_keys_values(parameter_setup):
    parameter_1, parameter_2 = parameter_setup

    def checks_same_keys_values(parameter):
        prices_to_int_dict, int_to_prices_dict = gen_price_combination_byte_mappings(
            parameter
        )

        # Check if both have the same keys/values
        for prices_string in prices_to_int_dict.keys():
            price_state_string_split = [item for item in prices_string.split("_") if item != '']
            price_state = np.array([eval(i) for i in price_state_string_split])
            assert any(
                [
                    np.array_equal(price_state, price_array)
                    for price_array in int_to_prices_dict.values()
                ]
            )

        for int_state in int_to_prices_dict.keys():
            assert int_state in prices_to_int_dict.values()

    # Check for both parameter setups
    checks_same_keys_values(parameter_1)
    checks_same_keys_values(parameter_2)


def test_gen_price_combination_byte_mappings_mapping_and_back(parameter_setup):
    parameter_1, parameter_2 = parameter_setup

    def check_mapping_and_back(parameter):
        prices_to_int_dict, int_to_prices_dict = gen_price_combination_byte_mappings(
            parameter
        )

        for int_state in int_to_prices_dict.keys():
            price_state_inital = int_to_prices_dict[int_state]
            int_state_reversed = prices_to_int_dict[
                _flatten_array_to_str(price_state_inital)
            ]
            assert int_state_reversed == price_state_to_int_state(
                price_state=price_state_inital, prices_to_int_dict=prices_to_int_dict
            )
            assert int_state_reversed == int_state

        for price_state_string in prices_to_int_dict.keys():
            price_state_string_split = [item for item in price_state_string.split("_") if item != '']
            price_state = np.array([eval(i) for i in price_state_string_split])
            int_state_initial = prices_to_int_dict[price_state_string]
            price_state_reversed = int_to_prices_dict[int_state_initial]

            assert np.array_equal(price_state_reversed, price_state)
            assert np.array_equal(
                price_state_reversed,
                int_state_to_price_state(
                    int_state=int_state_initial, int_to_prices_dict=int_to_prices_dict
                ),
            )

    check_mapping_and_back(parameter_1)
    check_mapping_and_back(parameter_2)


def test_gen_price_combination_byte_always_same(parameter_setup):
    parameter_1, parameter_2 = parameter_setup

    def check_always_same(parameter):
        (
            prices_to_int_dict_1,
            int_to_prices_dict_1,
        ) = gen_price_combination_byte_mappings(parameter)

        (
            prices_to_int_dict_2,
            int_to_prices_dict_2,
        ) = gen_price_combination_byte_mappings(parameter)

        assert prices_to_int_dict_1 == prices_to_int_dict_2
        # dict because they are numba typeddict
        np.testing.assert_equal(dict(int_to_prices_dict_1), dict(int_to_prices_dict_2))

    check_always_same(parameter_1)
    check_always_same(parameter_2)
