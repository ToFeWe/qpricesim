"""

A module that defines the QLearning Agent for the pricing game as a class.
Note that we have a numba version (for speed) which inherits everything from
QLearningAgentBase.
"""
import numpy as np
from numba import float64
from numba import int64
from numba import njit
from numba.experimental import jitclass

from .utils_q_learning import numba_argmax
from .utils_q_learning import numba_max

HUMAN_Q_MATRIX_2_FIRMS = np.array([
          [554.78, 597.65, 0.00, 2059.27, 1388.73, 0.00],
          [554.78, 601.88, 592.73, 568.89, 436.86, 0.00],
          [554.78, 602.65, 1100.62, 1019.52, 436.86, 0.00],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [2200.29, 1224.96, 1143.05, 568.89, 2341.10, 0.00],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [1357.29, 604.43, 1152.56, 0.00, 718.02, 228.00],
          [554.78, 603.05, 588.84, 578.89, 450.77, 228.00],
          [0.00, 610.88, 598.54, 568.89, 443.92, 1845.87],
          [0.00, 613.01, 1096.90, 618.89, 456.86, 0.00],
          [1357.29, 1209.33, 1626.40, 1988.56, 2316.65, 0.00],
          [0.00, 1732.00, 674.86, 838.93, 436.86, 0.00],
          [554.78, 1206.96, 1092.62, 568.89, 718.02, 0.00],
          [554.78, 611.37, 600.57, 591.39, 445.43, 1587.18],
          [441.74, 625.81, 1078.42, 1031.26, 718.02, 1587.18],
          [0.00, 627.33, 1090.20, 1666.78, 718.02, 0.00],
          [0.00, 1217.33, 1623.70, 1983.56, 2304.25, 0.00],
          [0.00, 616.88, 619.73, 1700.21, 2483.04, 0.00],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [554.78, 607.73, 618.36, 578.89, 448.86, 0.00],
          [554.78, 631.53, 1089.53, 1054.59, 718.02, 0.00],
          [0.00, 616.88, 1135.55, 1717.64, 1406.84, 0.00],
          [554.78, 639.77, 1635.97, 2002.43, 2316.22, 2223.21],
          [0.00, 0.00, 704.86, 1763.21, 1388.73, 0.00],
          [0.00, 649.01, 0.00, 1019.52, 2316.10, 0.00],
          [554.78, 609.62, 621.87, 576.39, 451.03, 0.00],
          [441.74, 621.88, 1087.50, 1697.04, 735.93, 228.00],
          [2200.29, 639.01, 1128.05, 1739.55, 1440.25, 0.00],
          [2200.29, 1222.35, 1637.00, 2025.22, 2359.24, 2223.21],
          [0.00, 0.00, 0.00, 0.00, 2379.10, 2223.21],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.00, 634.60, 574.73, 0.00, 0.00, 228.00],
          [0.00, 628.60, 604.73, 1718.21, 0.00, 1845.87],
          [0.00, 634.60, 0.00, 2029.27, 718.02, 1587.18],
          [0.00, 0.00, 1655.47, 2059.27, 2379.56, 2223.21],
          [0.00, 0.00, 0.00, 0.00, 2361.10, 0.00]])


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

    def __init__(self, alpha, epsilon, discount, n_actions, n_states, use_human_q_matrix=False):

        self.n_actions = n_actions
        self.n_states = n_states

        self._qvalues = np.random.rand(self.n_states, self.n_actions)
        # Add the human q-matrix to the q-values
        if use_human_q_matrix:
            self._qvalues += HUMAN_Q_MATRIX_2_FIRMS
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
    ("n_actions", int64),
    ("n_states", int64),
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
