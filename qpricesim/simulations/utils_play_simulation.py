"""

A module with different helper functions to play for a certain amount of
periods and to induce a deviation in the market.
"""

import numpy as np

from qpricesim.simulations.utils_simulation import (
    concatenate_new_price_state,
)
from qpricesim.simulations.utils_simulation import (
    price_state_to_int_state,
)


def play_with_deviation(
    parameter, all_agents, prices_to_int_dict, possible_prices, initial_price_state):
    """
    Starting from the state of convergence the agents play in the market
    for a certain amount of periods. Then, an exogenous deviation is induced
    in the sense that the first agent in the list *all_agents* deviates by
    playing a price one step (one integer) below the price he would have played.
    Then, from this new deviation price state, the agents continue playing
    using the learned limit strategies.

    Args:
        parameter (dict): Parameter for the deviation simulation
        all_agents (list): List of QLearningAgents
        prices_to_int_dict (dict): Mapping dict from price to index states
        possible_prices (array): Array of possible prices
        initial_price_state (array): Price state in which the simulation starts.

    Returns:
        array: Array with all simulated prices states. First state is the
               state of convergence.
               Note that the array is of shape
               (parameter["n_play_periods"], n_agents).
    """
    #  Periods before deviation + 1 deviation period + state of convergence
    periods_after_deviation = (
        parameter["n_play_periods"] - parameter["periods_before_deviation"] - 1 - 1
    )

    # Play rounds after the initial price state and before the deviation
    price_seq_inital = play_n_periods(
        n_periods=parameter["periods_before_deviation"],
        all_agents=all_agents,
        prices_to_int_dict=prices_to_int_dict,
        possible_prices=possible_prices,
        start_price_state=initial_price_state,
    )
    # Induce one period deviation
    # If the last price was already the lowest price possible
    # there is no possible deviation.

    # Price state that would have been played in the deviation
    # period if there would be no induced deviation
    price_state_before_deviation = price_seq_inital[-1, :]
    int_state_before_deviation = price_state_to_int_state(
        price_state=price_state_before_deviation, prices_to_int_dict=prices_to_int_dict
    )

    hypothetical_prices = play_period(
        all_agents=all_agents,
        possible_prices=possible_prices,
        int_state=int_state_before_deviation,
    )

    # Take the prices that would have been played and induce a deviation.
    # Note that the agent can only deviate if a smaller prices exists.
    n_agents = len(all_agents)
    if hypothetical_prices[0] == possible_prices.min():
        deviation_price_state = hypothetical_prices
    else:
        deviation_price_state = hypothetical_prices - np.array(
            [1] + [0] * (n_agents - 1)
        )

    # Continue to play for the remaining periods
    price_seq_after = play_n_periods(
        n_periods=periods_after_deviation,
        all_agents=all_agents,
        prices_to_int_dict=prices_to_int_dict,
        possible_prices=possible_prices,
        start_price_state=deviation_price_state,
    )
    out_array = np.vstack(
        (initial_price_state, price_seq_inital, deviation_price_state, price_seq_after)
    )
    return out_array

def play_period(all_agents, possible_prices, int_state):
    """

    Simulate a single period, starting at *int_state* and returns
    an array with new prices.

    Args:
        all_agents (list): List of QLearningAgents
        possible_prices (array): Array of possible prices in the market
                                 Note that it is used to map the integer
                                 action representation to a price (the index).
        int_state (integer): Integer/index representation of the state from which
                             the agents play the round.

    Returns:
        array: Array of prices played in the given round. Ordering is the same
               as given by the *all_agents* list.
    """
    n_agents = len(all_agents)

    new_prices = np.empty(n_agents, dtype=int)
    for id_agent, agent in enumerate(all_agents):
        int_action = agent.get_best_action(int_state)
        new_prices[id_agent] = possible_prices[int_action]
    return new_prices


def play_n_periods(
    all_agents, possible_prices, prices_to_int_dict, n_periods, start_price_state
):
    """
    Simulate *n_periods* of market interaction starting from state *start_price_state*
    (price state) and returns all simulated price states.


    Args:
        all_agents (list): List of QLearningAgents
        possible_prices (array): Array of possible prices
        prices_to_int_dict (dict): Mapping dict from price to int states
        n_periods (integer): Number of periods to simulate
        start_price_state (array): Initial price state from which to simulate.
                                   In all consecutive rounds the respective
                                   continuation state will be used.

    Returns:
        array: Array with all simulated prices states. Note that the shape of
               the array is (n_periods, n_agents).
    """
    n_agents = len(all_agents)

    # Define output array
    simulated_price_states = np.empty((n_periods, n_agents), dtype=int)

    # Get the index/integer representation of the initial price state
    int_state = price_state_to_int_state(
        price_state=start_price_state, prices_to_int_dict=prices_to_int_dict
    )
    price_state = start_price_state

    # Loop over all period which should be simulated!
    for period in range(n_periods):
        new_prices = play_period(
            all_agents=all_agents, possible_prices=possible_prices, int_state=int_state
        )
        # Concatenate the old and new prices to the new price state
        # Note that strictly speaking this is only really relevant
        # if the memory is greater than 1.
        price_state = concatenate_new_price_state(
            old_price_state=price_state, new_prices=new_prices, n_agent=n_agents
        )
        # Get integer state for the next period
        int_state = price_state_to_int_state(
            price_state=price_state, prices_to_int_dict=prices_to_int_dict
        )
        simulated_price_states[period, :] = price_state
    return simulated_price_states


def play_without_deviation(
    parameter, all_agents, prices_to_int_dict, possible_prices, initial_price_state
):
    """
    Simulate the market with all agents starting from the *initial_price_state*.

    Args:
        parameter (dict): Dictionary with the deviation simulation parameter.
        all_agents (list): List of QLearningAgents
        prices_to_int_dict (dict): Mapping dict from price to index states
        possible_prices (array): Array of possible prices in the market.
        initial_price_state (array): Price representation of the state of
                                     convergence.

    Returns:
        array: Array with all simulated prices states. First state is the
               state of convergence.
               Note that the array is of shape
               (parameter["n_play_periods"], n_agents).
    """
    # First price states in the sequence are the initial price state
    periods_after_inital = parameter["n_play_periods"] - 1
    price_seq = play_n_periods(
        n_periods=periods_after_inital,
        all_agents=all_agents,
        prices_to_int_dict=prices_to_int_dict,
        possible_prices=possible_prices,
        start_price_state=initial_price_state,
    )
    return np.vstack((initial_price_state, price_seq))