"""

A module which collects different function to run a market training
simulation with Q-learning agents.
"""
import numpy as np
from numba import njit
from numba.typed import List

from ..model_code.economic_environment import calc_reward
from ..model_code.economic_environment import calc_winning_price
from ..model_code.QLearningAgent import QLearningAgent
from .utils_simulation import best_action_same
from .utils_simulation import calc_epsilon
from .utils_simulation import concatenate_new_price_state
from .utils_simulation import gen_possible_prices
from .utils_simulation import gen_price_combination_byte_mappings
from .utils_simulation import parameter_to_typed_int_dict
from .utils_simulation import price_state_to_int_state
from .utils_simulation import set_numba_seed


def train_agents(parameter, random_seed):
    """

    The main function to run the simulation.
    The simulation parameters are specified in *parameter*.

    Importantly, we distinguish between price_states/states and action/prices.
    A price state is an array of prices and represent the state in the economic sense.
    The state itself is an unique integer, which corresponds to the column in the Q-matrix.
    To get the price state from the state integer use the function *price_state_to_int_state*
    the the corresponding mapping dict.
    The same idea is used for actions. The integer action is the row in the q-matrix.
    To get the price action use price_array[action].

    We use this logic to make the representation within the agent simple but still
    have prices to calculate the economic profit.

    *random_seed* is an integer value.


    Args:
        parameter (dict): Parameter dict as described as some place else TODO.
        random_seed (int): Random seed

    Returns:
        tuple: bool_convergence, state, all_agents

               bool_convergence (Boolean): True if the simulation converged
               state (integer): Integer state upon convergence
               all_agents (TypedList): TypedList(numba) of all Q-learner agents upon convergence.
                                       Note that you cannot pickle those as they are jitclasses
                                       but they can be exported by using the QLearningAgentBase
                                       class.
    """
    # Set the seed for this simulation run.
    # Will be the same as the PBS job index in my case.
    set_numba_seed(random_seed)

    # Create the agents
    all_agents, prices_to_int_dict, price_array = init_agents(parameter=parameter)

    # We start at the minimal possible price (initial state)
    # This choice is arbitrary
    price_state = np.array(
        [parameter["min_price"]] * parameter["n_agent"] * parameter["k_memory"],
        dtype=np.int64,
    )

    # Transform the price state to a simple integer representation
    state = price_state_to_int_state(
        price_state=price_state, prices_to_int_dict=prices_to_int_dict
    )

    # Some parameters to variables
    beta = parameter["beta_decay"]
    n_agent = parameter["n_agent"]

    # Create typeddict for numba routine
    # This will have the parameters needed for the training_iteations()
    parameter_typed = parameter_to_typed_int_dict(parameter)

    # Run the training iteration till convergence is acheived
    bool_convergence, state, all_agents, t_period = training_iterations(
        price_state=price_state,
        state=state,
        all_agents=all_agents,
        parameter_typed=parameter_typed,
        prices_to_int_dict=prices_to_int_dict,
        price_array=price_array,
        beta=beta,
        n_agent=n_agent,
    )

    # In order to check convergence, we compare if the array of best response did change
    # compared to the last training iteration.
    if bool_convergence:
        print(
            "We converged for the agent with beta={} and alpha={}".format(
                parameter["beta_decay"], parameter["learning_rate"]
            )
        )
        print(
            "Last state for agent with beta={} and alpha={} = {}".format(
                parameter["beta_decay"], parameter["learning_rate"], state
            )
        )
        print(f"It took us t={t_period} training iterations to converge")

    return bool_convergence, state, all_agents


@njit
def training_iterations(
    price_state,
    state,
    all_agents,
    parameter_typed,
    prices_to_int_dict,
    price_array,
    beta,
    n_agent,
):
    """
    A function to run the simulation for the specified number of
    periods or upon convergence.

    Args:
        price_state (array): Initial state as prices
        state (integer): Initial state in the integer representation
        all_agents (TypedList): TypedList(numba) with all QLearningAgents
        parameter_typed (TypedDict): TypedDict(numba) with the model parameters
                                     for the simulation.
                                     key (unicode): key-word
                                     value (int64): values
        prices_to_int_dict (TypedDict): Dictionary to map prices (array) to an integer state
        price_array (array): Array of possible prices
        beta (float): Decay parameter
        n_agent (integer): Number of agents

    Returns:
        tuple: bool_convergence, state, all_agents, t_period

               bool_convergence(bool): True if we converged
               state(integer): Integer representation of the last state
               all_agents(TypedList): All QLearningAgents
               t_period(integer): Number of learning periods
    """
    # Convergence checker
    convergence_counter = 0
    bool_convergence = False

    # Training loop
    t_period = 0
    while t_period < parameter_typed["learning_iterations"]:
        # In each training iteration, we take a single update for each agent
        all_agents, state, price_state, same_best_actions = single_iteration(
            past_price_state=price_state,
            past_state=state,
            all_agents=all_agents,
            parameter_typed=parameter_typed,
            t_period=t_period,
            prices_to_int_dict=prices_to_int_dict,
            price_array=price_array,
            beta=beta,
            n_agent=n_agent,
        )
        # Increase counter
        t_period += 1
        # In order to check convergence, we compare if the array of best response did change
        # compared to the last training iteration.
        if same_best_actions:
            convergence_counter += 1

            # Break the loop if we converged successfully
            if convergence_counter >= parameter_typed["rounds_convergence"]:
                bool_convergence = True
                break
        else:
            convergence_counter = 0

    return bool_convergence, state, all_agents, t_period


@njit
def single_iteration(
    past_price_state,
    past_state,
    all_agents,
    parameter_typed,
    t_period,
    prices_to_int_dict,
    price_array,
    beta,
    n_agent,
):
    """
    A function to take a single training step for all agents.
    TODO: Lots of arguments, as I want to avoid using gloabls.
    Maybe use a Parameter object here?

    Args:
        past_price_state (array): The past state as prices. We use this to create the new price
                                  state (if memory greater one) and map this into a proper state.
        past_state (array): Past state of the agent.
        all_agents (TypedList): List of all agents.
        parameter_typed (TypedDict): Parameter dict in as numba typed (TODO)
        t_period (int): The training period we are currently in. Used for the decay of beta.
        prices_to_int_dict (TypedDict): Dictionary to map prices (array) to an integer state
        price_array (array): Array of possible prices
        beta (float): Decay parameter
        n_agent (integer): Number of agents

    Returns:
        tuple: all_agents, new_state, new_price_state, same_actions_bool

               all_agents (TypedList): List of trained agents after another iteration
               new_state (int): Current integer state representation
               new_price_state (array): Current price state
               same_actions_bool (Boolean): True if the best actions did not change this period
    """

    # Initialize empty arrays for the output of the run:
    # Best actions before/after the update for the state for the state
    # that has changed.
    old_best_actions_iteration = np.empty(n_agent, dtype=np.int64)
    new_best_actions_iteration = np.empty(n_agent, dtype=np.int64)

    # Actually picked action in the state
    new_actions_iteration = np.empty(n_agent, dtype=np.int64)

    # Reward in the state
    rewards_iteration = np.empty(n_agent)

    # Get actions for each agent
    # We have to do this before the update loop, given that
    # we need all prices to calculate the rewards for each agent.
    for i_agent in range(n_agent):
        chosen_action, best_action = all_agents[i_agent].get_action(past_state)
        new_actions_iteration[i_agent] = chosen_action
        old_best_actions_iteration[i_agent] = best_action

    # Integer representation of the action is the index
    # of the price array.
    new_prices = price_array[new_actions_iteration]

    new_price_state = concatenate_new_price_state(
        old_price_state=past_price_state, new_prices=new_prices, n_agent=n_agent
    )
    new_state = price_state_to_int_state(
        price_state=new_price_state, prices_to_int_dict=prices_to_int_dict
    )

    # Exploration parameter epsilon for the following period
    # It is the same for all agents.
    new_epsilon = calc_epsilon(beta=beta, t=t_period)

    # Calculate the market price and the number of players that picked this
    # price.
    winning_price, n_winning_price = calc_winning_price(all_prices=new_prices)

    # Update loop for the agents.
    for i_agent in range(n_agent):
        # calculate the reward
        rewards_iteration[i_agent] = calc_reward(
            p_i=new_prices[i_agent],
            winning_price=winning_price,
            n_winning_price=n_winning_price,
            reservation_price=parameter_typed["reservation_price"],
            m_consumer=parameter_typed["m_consumer"],
        )

        # Update each agent
        all_agents[i_agent].update(
            past_state,
            new_actions_iteration[i_agent],
            rewards_iteration[i_agent],
            new_state,
        )

        # Save the best actions after the update:
        # This is needed to check convergence.
        new_best_actions_iteration[i_agent] = all_agents[i_agent].get_best_action(
            past_state
        )

        # Update the exploration parameter epsilon
        # for the next iteration.
        all_agents[i_agent].epsilon = new_epsilon

    # Check if the best action changed after the update.
    # This is, wrt to convergence, the only thing that could have changed as
    # Q-learning updates on cell at a time.
    same_actions_bool = best_action_same(
        old_best_response=old_best_actions_iteration,
        new_best_response=new_best_actions_iteration,
    )
    return all_agents, new_state, new_price_state, same_actions_bool


def init_agents(parameter):
    """

    Generates a list of agents given the simulation parameters. Furthermore,
    it returns a dictionary, that is used to map an array of prices to integer
    states, and a array of possible prices.

    Args:
        parameter (dict): Parameter dict as described as some place else TODO

    Returns:
        tuple: agent_list, prices_to_int_dict, price_array

               agent_list (TypedList): TypedDict(numba) of the initalized agents
               prices_to_int_dict (TypedDict): TypedDict(numba) to map prices to integer states
               price_array (array): Array of possible prices
    """

    # Create numba typed list to be used in njit function
    agent_list = List()
    n_agent = parameter["n_agent"]

    price_array = gen_possible_prices(parameter)
    prices_to_int_dict, _ = gen_price_combination_byte_mappings(parameter=parameter)

    for _ in range(n_agent):
        current_agent = QLearningAgent(
            alpha=parameter["learning_rate"],
            epsilon=parameter["exploration_rate"],
            discount=parameter["discount_rate"],
            n_actions=price_array.shape[0],
            n_states=len(prices_to_int_dict.keys()),
        )
        agent_list.append(current_agent)

    return agent_list, prices_to_int_dict, price_array
