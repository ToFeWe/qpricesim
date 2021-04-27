"""

Collection of functions to calculate performance measures for the
agents upon convergence.
"""
import numpy as np

from ..model_code.economic_environment import calc_reward
from ..model_code.economic_environment import calc_winning_price
from .utils_performance import best_response_share
from .utils_performance import check_if_nash_equilibrium
from .utils_performance import get_average_profitability
from .utils_performance import get_delta
from .utils_performance import get_specific_profitability
from .utils_performance import get_weighted_profitability
from .utils_performance import sort_list
from .utils_simulation import concatenate_new_price_state
from .utils_simulation import gen_possible_prices
from .utils_simulation import gen_price_combination_byte_mappings
from .utils_simulation import int_state_to_price_state
from .utils_simulation import price_state_to_int_state


def get_all_prices(
    all_agents,
    parameter,
    prices_to_int_dict,
    int_to_prices_dict,
    price_set,
    state_of_convergence,
):
    """
    A function to simulate the prices for a given market starting
    from the state of convergence.

    Args:
        all_agents (list): List of trained agents
        parameter (dict): Explain somewhere else TODO
        prices_to_int_dict (dict): Price state representation (as string) to the index/integer
                                   representation of the price state.
        int_to_prices_dict (dict): Index/integer representation of the state to the price state
                                   representation.
        price_set (array): Array of all prices that can be played by the agent.
        state_of_convergence (integer): Integer/index representation of the state of convergence

    Returns:
        array: Prices in the rounds after convergence, shape = (n_rounds,n_agents)
    """
    state = state_of_convergence
    price_state = int_state_to_price_state(
        int_state=state_of_convergence, int_to_prices_dict=int_to_prices_dict
    )
    n_agent = len(all_agents)
    rounds = parameter["avg_price_rounds"]
    all_prices = np.empty((rounds, n_agent), dtype=np.int64)

    for n_round in range(rounds):
        new_actions = np.empty(n_agent, dtype=np.int64)
        for i_agent in range(n_agent):
            best_action = all_agents[i_agent].get_best_action(state)
            new_actions[i_agent] = best_action

        new_prices = price_set[new_actions]

        # Create new state for the next iteration.
        price_state = concatenate_new_price_state(
            old_price_state=price_state, new_prices=new_prices, n_agent=n_agent
        )
        state = price_state_to_int_state(
            price_state=price_state, prices_to_int_dict=prices_to_int_dict
        )

        # Save the prices
        all_prices[n_round, :] = new_prices

    return all_prices


def get_transition_arrays(
    all_trained_competitors,
    parameter,
    position_agent,
    position_list_competitors,
    prices_to_int_dict,
    int_to_prices_dict,
    price_set,
):
    """
        Create two arrays which tells you for each state action combination
    what the reward is and what the next state is.

    The arrays have the following logic:

    profit_array[state, action] -> Reward from picking a in s
    next_state_array[state, action] -> Following state when picking a in s


    Note that each agent assumes his own price to be in a specific position
    of the price state. For instance, the first agent in the list of all agents
    assumes to be in position zero, the second in position 1 etc.

    We will account for this by the following method:
    1. We will create a *joint_position_list* which has the logic
       [position of the agent] + [ordered positions of all_trained_competitors].

       Example for the second agent with two competitors:
       joint_position_list = [1, 0, 2]

    2. Get the new prices for the given state for all competitors and
       combine it with the price from the agent. Save it to the list
       *joint_new_prices*.

       Example for the second agent with two competitors:
       joint_position_list = [p_1, p_0, p_2]

    3. Reorder the list *joint_new_prices* according to the correct positions
       of the prices that the agents assumed during traing using the *joint_position_list*.

       Example for the second agent with two competitors:
       new_prices_ordered = [p_0, p_1, p_2]

    4. If needed, in case of memory greater than 1, concate the *new_prices_ordered* and
       the part of the old_state, to get the new price state. If the memory is 1,
       *new_prices_ordered* is the new price state.


    Args:
        all_trained_competitors (list): List of QLearningAgents upon convergence
        parameter (dict): Parameter dictionary as described somewhere else TODO
        position_agent (integer): Position of the agent for which we derive the
                                  optimality measures in the list of *all* agents.
                                  This is needed as the position of the own price
                                  in the price state matters.
        position_list_competitors (list): List of positions in that each competitor has
                                          in the list of *all* agents.
                                          Here the order of this position list corresponds
                                          to the list of *all_trained_competitors*.
                                          Thus, for instance, position_list_competitors[0]
                                          tell you the position of the agent in
                                          all_trained_competitors[0] in the list of all
                                          agents.
        prices_to_int_dict (dict): Price state representation (as string) to the index/integer
                                   representation of the price state.
        int_to_prices_dict (dict): Index/integer representation of the state to the price state
                                   representation.
        price_set (array): Array of all prices that can be played by the agent.

    Returns:
        tuple: profit_array, next_state_array

               profit_array (array): Array which tell you for each state-action combination
                                     what the profit is given that the competitors
                                     play their limit strategy.
                                     Rows correspond to index representation of the actions.
                                     Columns correspond to index representation of the states.
               next_state_array (array): Array which tell you for each state-action combination
                                         what the continuation state is given that the competitors
                                         play their limit strategy.
                                         Rows correspond to index representation of the actions.
                                         Columns correspond to index representation of the states.

    """
    n_states = len(prices_to_int_dict.keys())
    n_actions = price_set.shape[0]
    n_agent = parameter["n_agent"]

    # A list of the positions of each agent in the state variable
    # The logic corresponds to [AGENT] + [all_trained_competitors]
    joint_position_list = [position_agent] + position_list_competitors

    profit_array = np.empty([n_states, n_actions])
    next_state_array = np.empty([n_states, n_actions], dtype=int)

    # Loop over all states
    for int_state, price_state in int_to_prices_dict.items():
        # Derive the deterministic action for the competitors for
        # the current state.
        all_prices_trained_competitors = []

        for trained_competitor in all_trained_competitors:
            int_action_trained = trained_competitor.get_best_action(int_state)
            all_prices_trained_competitors.append(price_set[int_action_trained])

        # For each state, get the reward and cont. state for each action
        for int_action in range(n_actions):
            price_action = price_set[int_action]

            # Create the new part of the price state variable based on the positions of the
            # agents.
            joint_new_prices = [price_action] + all_prices_trained_competitors
            new_prices_ordered = np.array(
                sort_list(
                    list_to_sort=joint_new_prices, sort_by_list=joint_position_list
                )
            )
            # If we consider price states with a larger history,
            # we append the new prices to the old remaining part of
            # the price_state.
            new_price_state = concatenate_new_price_state(
                old_price_state=price_state,
                new_prices=new_prices_ordered,
                n_agent=n_agent,
            )
            new_int_state = price_state_to_int_state(
                price_state=new_price_state, prices_to_int_dict=prices_to_int_dict
            )

            # Calculate the reward...
            winning_price, n_winning_price = calc_winning_price(
                all_prices=new_price_state
            )

            reward_state = calc_reward(
                p_i=price_action,
                winning_price=winning_price,
                n_winning_price=n_winning_price,
                reservation_price=parameter["reservation_price"],
                m_consumer=parameter["m_consumer"],
            )

            # Save the continuation state and the reward to the arrays
            profit_array[int_state, int_action] = reward_state
            next_state_array[int_state, int_action] = new_int_state

    return profit_array, next_state_array


def get_optimal_q_matrix(
    profit_array, next_state_array, parameter, int_to_prices_dict, price_set
):
    """
    We derive the optimal q-matrix using the two arrays *profit_array*
    and *next_state_array*.

    - next_state_array[int_state, int_action] -> next state when picking the action
    *int_action* in the state *int_state.
    - profit_array[int_state, int_action] -> profit when picking the action
    *int_action* in the state *int_state.

    Note that those arrays are possible to derive because we consider the strategies
    of the competitors as fixed.

    We find the optimal q-matrix by iteration of the q-function until convergence.
    We use the simple iterative policy evaluation algortihms described in Sutton&Barto



    Args:
        profit_array (array): Array which tell you for each state-action combination
                              what the profit is given that the competitors
                              play their limit strategy.
                              Rows correspond to index representation of the actions.
                              Columns correspond to index representation of the states.
        next_state_array (array): Array which tell you for each state-action combination
                                  what the continuation state is given that the competitors
                                  play their limit strategy.
                                  Rows correspond to index representation of the actions.
                                  Columns correspond to index representation of the states.
        parameter (dict): Explanation TODO
        int_to_prices_dict (dict): Index/integer representation of the state to the price state
                                   representation.
        price_set (array): Array of possible prices the agents can pick

    Returns:
        arrays: Optimal Q-matrix of the agent given the assumption that the competitors
                play their limit strategy.
    """
    # Retrieve number of states and action
    n_states = len(int_to_prices_dict.keys())
    n_actions = price_set.shape[0]

    optimal_q_matrix = np.ones([n_states, n_actions])
    discount_rate = parameter["discount_rate"]

    iteration_converged = False
    threshold_q_matrix_change = parameter["Q_star_threshold"]
    # Q-Matrix iteration
    while iteration_converged is False:
        # Save old q value
        old_q_value = np.array(optimal_q_matrix)

        # Calculate new q value with numpy broadcasting
        # To keep in mind: old_q_value[next_state_array, :] selects for a given state-action
        # combination all continuation payoffs for the next state for all possible actions.
        # Then we take the maximum over the axis=2 to get the value of the next
        # state
        optimal_q_matrix = profit_array + discount_rate * np.max(
            old_q_value[next_state_array, :], axis=2
        )

        # Check if we converge after each iteration over the entire
        # state-action space
        delta = get_delta(old_q_value=old_q_value, new_q_value=optimal_q_matrix)
        if delta < threshold_q_matrix_change:
            iteration_converged = True
    return optimal_q_matrix


def get_performance_measures(all_agents, parameter, convergence_state):
    """
    Calculate the relevant performance measures for the paper.


    Args:
        all_agents (list): List of all QLearningAgents upon convergence
        parameter (dict): Explain TODO
        convergence_state (integer): Index/integer representation of the
                                     state of convergence.

    Returns:
        tuple: out_profitability_state, out_weighted_profitability,
               out_best_response_share, out_avg_profit, q_matrices_tuple_list

               out_profitability_state (array): Value/Profitability in the state
                                                of convergence for all agents.
               out_weighted_profitability (array): Weighted average profitabiltiy
               out_best_response_share (array): Share of states in which the agent
                                                plays a best reponse for all agents.
               out_avg_profit (array): Average profitability across all states
                                       for all agents.
               out_prices_upon_convergence (array): Prices upon convergence
               out_nash_equilibrium (array): If the agent played a NE in the state
                                             of convergence.
               q_matrices_tuple_list (tuple): Tuple where each element is again a
                                              tuple with the q_matrix of the optimal
                                              and the actual agent.
                                              (optimal_agent_q_matrix, agent_q_matrix)
    """
    prices_to_int_dict, int_to_prices_dict = gen_price_combination_byte_mappings(
        parameter
    )
    price_set = gen_possible_prices(parameter)

    # During training, the first agent assumed to be in the first position in the state
    # the second one in the second position and so on. We have to account for this
    # when deriving the optimaliy.
    all_positions = list(range(len(all_agents)))

    # For each agent, we will have the actual q_matrix and the optimal q_matrix.
    # Those will be saved to a list of tuples
    q_matrices_tuple_list = []

    # Define the output arrays
    out_profitability_state = np.empty(parameter["n_agent"])
    out_weighted_profitability = np.empty(parameter["n_agent"])
    out_best_response_share = np.empty(parameter["n_agent"])
    out_avg_profit = np.empty(parameter["n_agent"])
    out_nash_equilibrium = np.empty(parameter["n_agent"])

    # Iterate over the list of agents to retrieve the optimal
    # counterpart for each agent.
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

        # Extract the q-matrix from the actual agent and compute all outputs.
        agent_q_matrix = agent.get_qmatrix()
        q_matrices_tuple_list.append((optimal_agent_q_matrix, agent_q_matrix))

        out_profitability_state[pos] = get_specific_profitability(
            actual_agent_q_matrix=agent_q_matrix, state=convergence_state
        )
        out_weighted_profitability[pos] = get_weighted_profitability(
            optimal_agent_q_matrix=optimal_agent_q_matrix,
            actual_agent_q_matrix=agent_q_matrix,
        )
        out_best_response_share[pos] = best_response_share(
            optimal_agent_q_matrix=optimal_agent_q_matrix,
            actual_agent_q_matrix=agent_q_matrix,
        )
        out_avg_profit[pos] = get_average_profitability(
            actual_agent_q_matrix=agent_q_matrix
        )
        out_nash_equilibrium[pos] = check_if_nash_equilibrium(
            optimal_agent_q_matrix=optimal_agent_q_matrix,
            actual_agent_q_matrix=agent_q_matrix,
            state=convergence_state,
        )

    out_prices_upon_convergence = get_all_prices(
        all_agents=all_agents,
        parameter=parameter,
        prices_to_int_dict=prices_to_int_dict,
        int_to_prices_dict=int_to_prices_dict,
        price_set=price_set,
        state_of_convergence=convergence_state,
    )
    return (
        out_profitability_state,
        out_weighted_profitability,
        out_best_response_share,
        out_avg_profit,
        out_prices_upon_convergence,
        out_nash_equilibrium,
        q_matrices_tuple_list,
    )
