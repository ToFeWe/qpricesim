"""

A module that defines the economic environment the agents are interacting in.
"""
import numpy as np
from numba import njit


@njit
def calc_winning_price(all_prices):
    """
    Helper function that takes in the array of all prices in the market
    and returns the winning price and the number of firms that picked
    this winning price.



    Args:
        all_prices (array): Array of all prices in the given round in the market

    Returns:
        tuple: winning_price, n_winning_price

               winning_price (integer): Current market price
               n_winning_price (integer): Number of firms that played the market price
    """
    # Get winning price
    # Lowest price wins the market
    winning_price = np.min(all_prices)

    # Get the number of players that played the winning price
    n_winning_price = np.sum(np.where(all_prices == winning_price, 1, 0))
    return winning_price, n_winning_price


@njit
def calc_reward(p_i, winning_price, n_winning_price, reservation_price, m_consumer):
    """
    A function that calculates the reward given a simple Bertrand
    environment with homogenous goods.

    Use calc_winning_price() to retrieve winning_price, n_winning_price
    for the given market prices first.

    Args:
        p_i (integer): Price the agent picked in the given round
                       (Non-index reprenstation of the action).
        winning_price (integer): Market price
        n_winning_price (integer): Number of firms that played the market price
        reservation_price (integer): Maximal price the consumers are willing to pay
        m_consumer (integer): Number of consumers in the market

    Returns:
        float: Economics profit/reward for the agent in the given period
    """

    # If the agents charges a price above reservation price, he comes home with zero.
    # If he plays the winning price, he shares the market with the others who played
    # the winning price.
    # If his price is above the winning price, he also goes home with zero.
    if p_i > reservation_price:
        return 0
    elif p_i == winning_price:
        return (1 / n_winning_price) * p_i * m_consumer
    else:
        return 0
