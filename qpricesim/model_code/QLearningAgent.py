"""

A module that defines the QLearning Agent for the pricing game as a class.
Note that we have a numba version (for speed) which inherits everything from
QLearningAgentBase.
"""
import numpy as np
from numba import float64
from numba import int32
from numba import njit
from numba.experimental import jitclass

from .utils_q_learning import numba_argmax
from .utils_q_learning import numba_max


class QLearningAgentBase:
    """

    A simple Q-Learning Agent based on numpy. Actions and state are assumed
    to be represented by integer numbers/an index and corresponds to the
    respective rows / columns in the Q-Matrix.
    We assume that the agent can choose every action in every state.

    The random seed will be set by a helper function outside this class.


    Args:
        self.epsilon (float): Exploration probability
        self.alpha (float): Learning rate
        self.discount (float): Discount rate
        self.n_actions (int): Number of actions the agent can pick
    """

    def __init__(self, alpha, epsilon, discount, n_actions, n_states):

        self.n_actions = n_actions
        self.n_states = n_states

        self._qvalues = np.random.rand(self.n_states, self.n_actions)
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def set_qmatrix(self, new_matrix):
        self._qvalues = new_matrix

    def get_qvalue(self, state, action):
        """

        Returns the Q-value for the given state action

        Args:
            state (integer): Index representation of a state
            action (integer): Index representation of an action

        Returns:
            float: Q-value for the state-action combination
        """
        return self._qvalues[state, action]

    def set_qvalue(self, state, action, value):
        """Sets the Qvalue for [state,action] to the given value

        Args:
            state (integer): Index representation of a state
            action (integer): Index representation of an action
            value (float): Q-value that is being assigned
        """
        self._qvalues[state, action] = value

    def get_value(self, state):
        """

        Compute the agents estimate of V(s) using current q-values.

        Args:
            state (integer): Index representation of a state

        Returns:
            float: Value of the state
        """
        value = numba_max(
            self._qvalues[
                state,
            ]
        )

        return value

    def get_qmatrix(self):
        """
        Returns the qmatrix of the agent


        Returns:
            array (float): Full Q-Matrix
        """

        return self._qvalues

    def update(self, state, action, reward, next_state):
        """
        Update Q-Value:
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))

        Args:
            state (integer): Index representation of the current state (Row of the Q-matrix)
            action (integer): Index representation of the picked action (Column of the Q-matrix)
            reward (float): Reward for picking from picking the action in the given state
            next_state (integer): Index representation of the next state (Column of the Q-matrix)
        """
        # Calculate the updated Q-value
        c_q_value = (1 - self.alpha) * self.get_qvalue(state, action) + self.alpha * (
            reward + self.discount * self.get_value(next_state)
        )

        # Update the Q-values for the next iteration
        self.set_qvalue(state, action, c_q_value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).

        Args:
            state (integer): Index representation of the current state (Row of the Q-matrix)

        Returns:
            integer: Index representation of the best action (Column of the Q-matrix)
                     for the given state (Row of the Q-matrix)
        """

        # Pick the Action (Row of the Q-matrix) with the highest q-value
        best_action = numba_argmax(self._qvalues[state, :])
        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we take a random action.

        Returns both, the chosen action (with exploration) and the best action (argmax).
        If the chosen action is the same as the best action, both returns will be
        the same.


        Args:
            state (integer): Integer representation of the current state (Row of the Q-matrix)

        Returns:
            tuple: chosen_action, best_action

                   chosen_action (integer): Index representation of the acutally picked action
                                            (Column of the Q-matrix)
                   best_action (integer): Index representation of the current best action
                                          (Column of the Q-matrix) in the given state.
        """
        # agent parameters:
        epsilon = self.epsilon
        e_threshold = np.random.random()

        # Get the best action.
        best_action = self.get_best_action(state)

        if e_threshold < epsilon:
            # In the numpy.random module randint() is exclusive for the upper
            # bound and inclusive for the lower bound -> Actions are array
            # indices for us.
            chosen_action = np.random.randint(0, self.n_actions)
        else:
            chosen_action = best_action
        return chosen_action, best_action


spec = [
    ("n_actions", int32),
    ("n_states", int32),
    ("_qvalues", float64[:, :]),
    ("alpha", float64),
    ("epsilon", float64),
    ("discount", float64),
]


@jitclass(spec)
class QLearningAgent(QLearningAgentBase):
    """
    Wrapper class to create a jitclass for the QLearningAgent.

    Not that this class cannot be serialized. Hence, if you want
    to save the trained agent as a pickle file, use the base class.

    Note that for the random seed to work, you need to do it in
    a njit wrapper function. From the numba documentation:
    "Calling numpy.random.seed() from non-Numba code (or from object mode code)
    will seed the Numpy random generator, not the Numba random generator."


    """


def jitclass_to_baseclass(agent_jit):
    """
    A helper function to create a new QLearningAgentBase
    object from the jitclass equivalent. This is needed
    as we cannot serialize jitclasses in the current
    numba version.
    The function takes all parameters from the QLearningAgent
    *agent_jit* and rewrites it to a new QLearningAgentBase
    object.

    Args:
        agent_jit (QLearningAgent): jitclass instance of agent

    Returns:
        QLearningAgentBase: Serializable version of the agent
    """
    agent_nojit = QLearningAgentBase(
        alpha=agent_jit.alpha,
        epsilon=agent_jit.epsilon,
        discount=agent_jit.discount,
        n_actions=agent_jit.n_actions,
        n_states=agent_jit.n_states,
    )
    agent_nojit.set_qmatrix(new_matrix=agent_jit.get_qmatrix())
    return agent_nojit
