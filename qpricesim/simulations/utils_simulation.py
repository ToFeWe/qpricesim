"""

Different helper function for the simulation with Q-learning agents in a Bertrand market.
"""
import math
from itertools import product

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict


@njit
def set_numba_seed(seed):
    """
    Wrapper function to create a random seed for
    numpy when used with numba.
    From the numba documentation:
    "Calling numpy.random.seed() from non-Numba code
    (or from object mode code) will seed the Numpy random
    generator, not the Numba random generator."


    Args:
        seed (integer): Random seed to be used in the simulation
    """
    np.random.seed(seed)


def parameter_to_typed_int_dict(parameter):
    """
    Helper function to create a typed dict with
    key:unicode and value:int64 with specific
    values from *parameter* to be used in njit
    functions.

    Args:
        parameter (dict): Parameter dictionary as explained elsewhere TODO

    Returns:
        typed-dict: [Numba typed dict
    """

    typed_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )

    typed_dict["learning_iterations"] = parameter["learning_iterations"]
    typed_dict["reservation_price"] = parameter["reservation_price"]
    typed_dict["m_consumer"] = parameter["m_consumer"]
    typed_dict["rounds_convergence"] = parameter["rounds_convergence"]

    return typed_dict


@njit
def concatenate_new_price_state(old_price_state, new_prices, n_agent):
    """
    Helper function that takes in the *old_price_state* and the *new_prices*
    to cut of the oldest part of the old prices and concatenate to the end
    the new prices for all agents.


    Args:
        old_price_state (array): Price state in the last period
        new_prices ([type]): Prices that are played in the given period
        n_agent ([type]): Number of agents in the market
                          Note that we need this in order to know,
                          how much of the old price state becomes irrelevant
                          for the new price state.
                          We do not need to know the memory if the inital state
                          was specified correctly.
    Returns:
        array: New price state
    """
    return np.concatenate((old_price_state[n_agent:], new_prices))


@njit
def best_action_same(old_best_response, new_best_response):
    """
    Helper function to check if the best responses of the agents for the old state changed
    after the update.

    Note that only the best action for the current state could have changed.
    Given the speed of the jit compiler but also its limitation, it turns out
    that it is still faster to check the whole array if the state-space is not
    too big.

    Args:
        old_best_response (array): Best reponses for all states before the update
        new_best_response (array): Best reponses for all states after the update

    Returns:
        Boolean: True if the best actions did NOT change after the update
    """
    return np.all(old_best_response == new_best_response)


def gen_possible_price_combination(parameter):
    """
    Generates a list with price states that can be visited by
    the agents, given the number of players, the memory and the action space.

    Args:
        parameter (dict): Parameter dictionary as explained elsewhere TODO

    Returns:
        list: List with all prices combinations that can appear in the market as a state.
              The price states are tuples here and NOT arrays like in the rest of the script.
    """
    n_agent = parameter["n_agent"]
    k_memory = parameter["k_memory"]
    price_set_list = list(gen_possible_prices(parameter=parameter))
    price_combinations_list = list(product(*[price_set_list] * n_agent * k_memory))

    return price_combinations_list


def gen_possible_prices(parameter):
    """
    Helper function to generate an array of
    possible prices the agents can chose given the
    assumed market setup.


    Args:
        parameter (dict): Parameter dictionary as explained elsewhere TODO

    Returns:
        array : All prices that can be picked in the market
    """
    return np.arange(
        parameter["min_price"],
        parameter["max_price"] + 1,
        parameter["step"],
        dtype=np.int64,
    )


@njit
def _flatten_array_to_str(array):
    """
    Helper function to reduce an array to a string
    to make it immutable.

    Args:
        array (array): Array to make to string

    Returns:
        string: String where each value from the array is appended to each other
    """
    s = ""
    for i in array:
        s += str(i)
    return s


def gen_price_combination_byte_mappings(parameter):
    """
    We map each price combination to a unique integer which represents the row in the q_matrix
    As the price combinations will be numpy arrays but should be keys, we transform it
    to a string to make them immutable.


    Then, when looking up the state integer for a price state, we always have to transform
    the price array to string as well.

    Furthermore, the function returns a mapping dict from the integer stage directly
    to a numpy array that represents the price state.

    When transforming state, always use the functions
    price_state_to_int_state() and int_state_to_price_state() with the
    corresponding mapping dictionaries.


    Args:
        parameter (dict): Parameter dictionary as explained elsewhere TODO

    Returns:
        tuple: prices_to_int_dict, int_to_prices_dict

               prices_to_int_dict (dict): Maps an arrays of prices (as string)
                                          to a unique index/integer representation
               int_to_prices_dict (dict): Maps a unique index/integer representation
                                          of a state to the corresponding price array.
    """
    price_combinations_list = gen_possible_price_combination(parameter)
    prices_to_int_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )
    int_to_prices_dict = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )
    for index_val in range(len(price_combinations_list)):
        price_state_array = np.array(price_combinations_list[index_val], dtype=np.int64)
        prices_to_int_dict[_flatten_array_to_str(price_state_array)] = index_val
        int_to_prices_dict[index_val] = price_state_array
    return prices_to_int_dict, int_to_prices_dict


@njit
def price_state_to_int_state(price_state, prices_to_int_dict):
    """
    Returns the integer state representation for the *price_state* given
    the mapping dict *prices_to_int_dict*.


    Args:
        price_state (array): Price state as an array
        prices_to_int_dict (dict): Mapping dict as returned by the function
                                   gen_price_combination_byte_mappings()

    Returns:
        integer: Index/integer representation of the state
    """
    return prices_to_int_dict[_flatten_array_to_str(price_state)]


@njit
def int_state_to_price_state(int_state, int_to_prices_dict):
    """
    Returns the price state representation for the *price_state* given
    the mapping dict *int_to_prices_dict*.


    Args:
        int_state (integer): Index/integer representation of the state
        int_to_prices_dict (dict): Mapping dict as returned by the function
                                   gen_price_combination_byte_mappings()

    Returns:
        array: Price state as an array
    """
    return int_to_prices_dict[int_state]


@njit
def calc_epsilon(beta, t):
    """
    Helper function that calculates the new exploration
    rate epsilon given the fixed deca parameter *beta*
    and the period indicator *t*.


    Args:
        beta (float): Decay parameter
        t (integer): Period indicator

    Returns:
        float: Exploration rate in the given period
    """
    return np.exp(-beta * t)
