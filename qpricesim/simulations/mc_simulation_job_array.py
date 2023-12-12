"""

A module with functions that will be called in PBS when running the simulation.
"""
from copy import deepcopy

import numpy as np

from ..model_code.QLearningAgent import jitclass_to_baseclass
from .agents_simulation import train_agents
from .performance import get_performance_measures
from .utils_simulation import gen_possible_price_combination


def run_specific_specification(beta, alpha, base_parameter, random_seed):
    """
    Runs a specific simulation for a fixed beta, alpha and random_seed.

    Args:
        beta (float): Decay parameter for the exploration rate
        alpha (float): Learning rate
        base_parameter (dict): Parameter dictionary as described somewhere else TODO
        random_seed (list): Random seed to be used

    Returns:
        list: Outcomes: profitability_state, weighted_profitability, best_response_share,
              avg_profit, state, trained_agents

              profitability_state (float): Value of the state of convergence
              weighted_profitability (float): Average profitability across all states
                                              weighted by the best_response_share
              best_response_share (float): Share of states in which the agent plays
                                           the best reponse (Optimality criterion).
              avg_profit (float): Average of the value of all states.
              prices_upon_convergence (float): Prices after convergence
              nash_equilibrium (boolean): True if the agents play jointly a Nash Equilibrium
              all_best_actions (array): Best actions for each agent in each state
              out_periods_shock (array): Shock to agent 1 in each period
              state (integer): Index representation of the state of convergence
              trained_agents (list): List of all QLearningAgents upon convergence
    """
    parameter = base_parameter
    parameter["beta_decay"] = beta
    parameter["learning_rate"] = alpha
    out = train_agents(parameter=parameter, random_seed=random_seed)

    # Unpack the output from training
    convergence_bool, state, trained_agents_jit = out

    # Transform jitclass QLearned to no-jitted version
    # we need it to be able to pickle it later.
    trained_agents = []
    for agent in trained_agents_jit:
        trained_agents.append(jitclass_to_baseclass(agent_jit=agent))

    convergence_info = (beta, alpha, convergence_bool)
    print("Did we converge?", convergence_bool)

    # Get the performance measures for the trained agents
    performance_measures = get_performance_measures(
        all_agents=trained_agents, parameter=base_parameter, convergence_state=state
    )
    profitability_state = performance_measures[0]
    weighted_profitability = performance_measures[1]
    best_response_share = performance_measures[2]
    avg_profit = performance_measures[3]
    prices_upon_convergence = performance_measures[4]
    nash_equilibrium = performance_measures[5]
    all_best_actions = performance_measures[6]
    out_periods_shock = performance_measures[7]

    return (
        profitability_state,
        weighted_profitability,
        best_response_share,
        avg_profit,
        prices_upon_convergence,
        nash_equilibrium,
        all_best_actions,
        out_periods_shock,
        state,
        trained_agents,
    )


# TODO: Could do integration test here..
def run_single_simulation(base_parameter, cases, job_array_index):
    """
    For for all agents, run simulations over the whole grid space fo alphas and betas.
    We use the *job_array_index* to create unique random seeds for all agents.



    Args:
        base_parameter (dict): TODO Explanation
        cases (dict): Dictionary that specifies the parameter grid which we search
        job_array_index (integer): Index of the job array from PBS. It is used to create
                                   a list of random seeds for all agents in the simulation
                                   which is also unique in the job array run.

    Returns:
        dict: Dictionary with all outputs

              Keys - values:
              state_profitability_array (array): Result of the grid simulation for the profitability
                                                 in the state of convergence.
              weighted_profitability_array (array): Result of the grid simulation for the weighted
                                                 average profitability.
              best_response_share_array (array): Result of the grid simulation for the best
                                                 response share (optimality).
              avg_profit_array (array): Result of the grid simulation for the average
                                        profitability.
              avg_price_array (array): Result of the grid simulation for the average
                                       prices after convergence.
              nash_equilibrium_array (array): Results of the grid simulation for the
                                              Nash Equilibrium analysis.
              all_best_actions_array (array): Best actions for each agent in each state
              periods_shock_array (array): Shock to agent 1 and then prices
              super_star_tuple (tuple): Tuple with the best agent in this simulation run
                                        according to the weighted average profitability

                                        trained_agents[0] (QLearningAgent): Trained super star
                                        alpha_grid[i_alpha] (float): Learning rate
                                        beta_grid[i_beta] (float): Decay parameter
                                        weighted_profitability (float): Weighted
                                                                        average profitability
                                        best_response_share (float): Best response share
                                                                     (Optimality)
                                        profitability_state (float): Profitability in the state
                                                                     of convergence
                                        avg_profit (float): Average profitability
                                        prices_upon_convergence (array): Price array upon convergence
                                        nash_equilibrium (boolean): True if both NE is played
                                        state (integer): Index representation of the state
                                                         of convergence
                                        random_seed (list): Random seed/job array index

    """

    print("Starting with Monte Carlo for the random seed \n", job_array_index)
    beta_grid = np.linspace(cases["beta_min"], cases["beta_max"], cases["grid_points"])
    alpha_grid = np.linspace(
        cases["alpha_min"], cases["alpha_max"], cases["grid_points"]
    )

    # output arrays
    weighted_profitability_array = np.empty(
        (cases["grid_points"], cases["grid_points"])
    )

    best_response_share_array = np.empty((cases["grid_points"], cases["grid_points"]))
    state_profitability_array = np.empty((cases["grid_points"], cases["grid_points"]))
    avg_profit_array = np.empty((cases["grid_points"], cases["grid_points"]))
    avg_price_array = np.empty((cases["grid_points"], cases["grid_points"]))
    nash_equilibrium_array = np.empty(
        (cases["grid_points"], cases["grid_points"]), dtype=bool
    )

    # Best action array is ndarray of size grid_points x grid_points x agents x states
    # We need to store the best action for each agent in each state for each alpha and beta
    n_states = len(gen_possible_price_combination(base_parameter))
    n_agents = base_parameter["n_agent"]
    all_best_actions_array = np.empty((cases["grid_points"], cases["grid_points"],
                                       n_agents, n_states), dtype=np.int8)
    
    # Shock array is ndarray of size grid_points x grid_points x periods x agents
    # Note that this is the other way around than the best action array (TODO: Maybe change this)
    # Furthermore, note that we only consider the shock to agent 1
    n_sim_play_periods = base_parameter["n_play_periods"]
    periods_shock_array = np.empty((cases["grid_points"], cases["grid_points"],
                                           n_sim_play_periods, n_agents), dtype=np.int8)
       

    # We search in each monte carlo simulation for the super star agent
    # This is done by looking for the agent with the highest average weighted
    # profitability
    current_max_profitability = 0

    # Tuple to save everything from the best performing agent
    out_super_star_tuple = ()

    # We use a deepcopy here to ensure proper
    # encapsulation if the simulation runs in parallel
    base_parameter_in = deepcopy(base_parameter)
    # Outer loop over the different alphas
    for i_alpha in range(cases["grid_points"]):
        for i_beta in range(cases["grid_points"]):
            # output from the simulation
            out = run_specific_specification(
                beta=beta_grid[i_beta],
                alpha=alpha_grid[i_alpha],
                base_parameter=base_parameter_in,
                random_seed=int(job_array_index),
            )

            # unpack output from the simulation
            (
                profitability_state,
                weighted_profitability,
                best_response_share,
                avg_profit,
                prices_upon_convergence,
                nash_equilibrium,
                all_best_actions,
                out_periods_shock,
                state,
                trained_agents,
            ) = out
            state_profitability_array[i_alpha, i_beta] = np.mean(profitability_state)
            best_response_share_array[i_alpha, i_beta] = np.mean(best_response_share)
            weighted_profitability_array[i_alpha, i_beta] = np.mean(
                weighted_profitability
            )
            avg_profit_array[i_alpha, i_beta] = np.mean(avg_profit)
            avg_price_array[i_alpha, i_beta] = np.mean(prices_upon_convergence)
            nash_equilibrium_array[i_alpha, i_beta] = all(nash_equilibrium)
            all_best_actions_array[i_alpha, i_beta, :, :] = all_best_actions
            periods_shock_array[i_alpha, i_beta, :, :] = out_periods_shock

            # Check if the current agent is the current super star
            # Note that we take the mean of all agents to
            # avoid wins due to asymmetric learning only
            mean_weighted_profitability = np.mean(weighted_profitability)
            if mean_weighted_profitability > current_max_profitability:
                out_super_star_tuple = (
                    trained_agents[0],
                    alpha_grid[i_alpha],
                    beta_grid[i_beta],
                    weighted_profitability,
                    best_response_share,
                    profitability_state,
                    avg_profit,
                    prices_upon_convergence,
                    nash_equilibrium,
                    state,
                    job_array_index,
                )
                current_max_profitability = mean_weighted_profitability

    # Output dict, which we return
    outputs = dict(
        state_profitability_array=state_profitability_array,
        weighted_profitability_array=weighted_profitability_array,
        best_response_share_array=best_response_share_array,
        avg_profit_array=avg_profit_array,
        avg_price_array=avg_price_array,
        nash_equilibrium_array=nash_equilibrium_array,
        all_best_actions_array=all_best_actions_array,
        periods_shock_array=periods_shock_array,
        super_star_tuple=out_super_star_tuple,
    )

    return outputs
