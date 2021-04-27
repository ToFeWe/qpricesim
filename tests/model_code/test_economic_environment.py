"""
Tests for the economic environment module.

"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from qpricesim.model_code.economic_environment import calc_reward
from qpricesim.model_code.economic_environment import calc_winning_price


@pytest.fixture
def setup():
    all_other_prices = [5, 5]
    reservation_price = 10
    m_consumer = 60
    return all_other_prices, reservation_price, m_consumer


def test_winning_price_1():
    all_prices = np.array([3, 4, 5])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    assert winning_price == 3 and n_winning_price == 1


def test_winning_price_2():
    all_prices = np.array([10, 10, 10, 10])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    assert winning_price == 10 and n_winning_price == 4


def test_winning_price_3():
    all_prices = np.array([1, 1, 10, 10, 2])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    assert winning_price == 1 and n_winning_price == 2


def test_calc_reward_all_same(setup):
    all_other_prices, reservation_price, m_consumer = setup
    price = 5
    all_prices = np.array(all_other_prices + [price])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    reward = calc_reward(
        p_i=price,
        winning_price=winning_price,
        n_winning_price=n_winning_price,
        reservation_price=reservation_price,
        m_consumer=m_consumer,
    )

    assert_array_almost_equal(reward, 100)


def test_calc_reward_own_above(setup):
    all_other_prices, reservation_price, m_consumer = setup
    price = 6
    all_prices = np.array(all_other_prices + [price])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    reward = calc_reward(
        p_i=price,
        winning_price=winning_price,
        n_winning_price=n_winning_price,
        reservation_price=reservation_price,
        m_consumer=m_consumer,
    )

    assert_array_almost_equal(reward, 0)


def test_calc_reward_own_below(setup):
    all_other_prices, reservation_price, m_consumer = setup
    price = 4
    all_prices = np.array(all_other_prices + [price])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    reward = calc_reward(
        p_i=price,
        winning_price=winning_price,
        n_winning_price=n_winning_price,
        reservation_price=reservation_price,
        m_consumer=m_consumer,
    )

    assert_array_almost_equal(reward, 240)


def test_calc_reward_all_above_reservation(setup):
    _, reservation_price, m_consumer = setup
    price = 11
    all_other_prices = [12, 13]
    all_prices = np.array(all_other_prices + [price])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    reward = calc_reward(
        p_i=price,
        winning_price=winning_price,
        n_winning_price=n_winning_price,
        reservation_price=reservation_price,
        m_consumer=m_consumer,
    )

    assert_array_almost_equal(reward, 0)


def test_calc_reward_all_zero(setup):
    _, reservation_price, m_consumer = setup
    price = 0
    all_other_prices = [0, 0]
    all_prices = np.array(all_other_prices + [price])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    reward = calc_reward(
        p_i=price,
        winning_price=winning_price,
        n_winning_price=n_winning_price,
        reservation_price=reservation_price,
        m_consumer=m_consumer,
    )

    assert_array_almost_equal(reward, 0)


def test_calc_reward_all_zero(setup):
    _, reservation_price, m_consumer = setup
    price = 0
    all_other_prices = [0, 0]
    all_prices = np.array(all_other_prices + [price])
    winning_price, n_winning_price = calc_winning_price(all_prices)
    reward = calc_reward(
        p_i=price,
        winning_price=winning_price,
        n_winning_price=n_winning_price,
        reservation_price=reservation_price,
        m_consumer=m_consumer,
    )

    assert_array_almost_equal(reward, 0)
