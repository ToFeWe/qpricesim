"""

A module with different helper functions to calculate the
performance measures.
"""
import numpy as np
from numba import njit


def get_weighted_profitability(optimal_agent_q_matrix, actual_agent_q_matrix):
    """
    Get the profitability, given by the average value over all states,
    weighted by the optimality of the agent.

    Args:
        optimal_agent_q_matrix (array): Q-Matrix of the optimal agent
        actual_agent_q_matrix (array): Q-Matrix of the acutal agent

    Returns:
        float: Weighted average profitability
    """
    avg_profit_actual = get_average_profitability(
        actual_agent_q_matrix=actual_agent_q_matrix
    )
    br_share = best_response_share(
        optimal_agent_q_matrix=optimal_agent_q_matrix,
        actual_agent_q_matrix=actual_agent_q_matrix,
    )
    return br_share * avg_profit_actual


def get_average_profitability(actual_agent_q_matrix):
    """
    Calculates the average profitability over the entire state space.

    Args:
        actual_agent_q_matrix (array): Q-Matrix of the acutal agent

    Returns:
        float: Average profitability
    """
    maxs_actual = actual_agent_q_matrix.max(axis=1)
    avg_profit = np.mean(maxs_actual)
    return avg_profit


@njit
def get_specific_profitability(actual_agent_q_matrix, state):
    """
    Calculate the profitability for the integer state *state*

    Args:
        actual_agent_q_matrix (array): Q-Matrix of the acutal agent
        state (integer): Integer/Index representation of a state

    Returns:
        float: Value/Profitability of the state
    """
    return np.max(actual_agent_q_matrix[state, :])


@njit
def get_delta(old_q_value, new_q_value):
    """
    Calculate the maximal absolute difference betweens two
    q-matricies. It is used to check convergences when deriving
    the optimal q-matrix.

    Args:
        old_q_value (array): Q-matrix of the last iteration
        new_q_value (array): Q-matrix in this iteration

    Returns:
        float: Maximal absolute change in the q-matrix within this iteration
    """
    return np.max(np.abs(np.subtract(old_q_value, new_q_value)))


def best_response_share(optimal_agent_q_matrix, actual_agent_q_matrix):
    """
    Function that calculates the share of states in which the agents best response
    is equivalent to the best response of the optimal agent. In other words, what is
    the share of NE it plays for all states. If the value is equal to one, it found
    a SPNE.

    Args:
        optimal_agent_q_matrix (array): Q-matrix of the optimal agent
        actual_agent_q_matrix (array): Q-matrix of the actual agent

    Returns:
        float: Fraction of state in which the actual agent has played a
               best reponse.
    """
    br_optimal_agent = np.argmax(optimal_agent_q_matrix, axis=1)
    br_actual_agent = np.argmax(actual_agent_q_matrix, axis=1)
    return np.mean(br_optimal_agent == br_actual_agent)


def check_if_nash_equilibrium(optimal_agent_q_matrix, actual_agent_q_matrix, state):
    """
    Function to check if the agent played a Nash equilibrium for the
    given *state*.

    Args:
        optimal_agent_q_matrix (array): Q-matrix of the optimal agent
        actual_agent_q_matrix (array): Q-matrix of the actual agent
        state (integer): Integer/Index representation of a state

    Returns:
        boolean: True if the agent played a NE
    """
    action_optimal_agent = np.argmax(optimal_agent_q_matrix[state, :])
    action_actual_agent = np.argmax(actual_agent_q_matrix[state, :])

    return action_optimal_agent == action_actual_agent


def sort_list(list_to_sort, sort_by_list):
    """
    Sort the list *list_to_sort* by the index list
    *sort_by_list*.


    Args:
        list_to_sort (list): List that should be sorted
        sort_by_list (list): List with indicies that are used to
                             sort *list_to_sort*

    Returns:
        list: Sorted list
    """
    zipped_pairs = zip(sort_by_list, list_to_sort)

    sorted_list = [x for _, x in sorted(zipped_pairs)]

    return sorted_list
