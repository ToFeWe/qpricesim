==========================
Q-Learner price simulation
==========================

A package to simulate q-learning agents in an economics environment with non-elastic demand.
The package is used in the following `working paper <https://tofewe.github.io/Algorithmic_and_Human_Collusion_Tobias_Werner.pdf>`_
The paper explains the simulations and the parameter that are used in great detail.


The package uses `numba <https://numba.pydata.org/>`_ to achieve a fast run time.
It is designed to be used in large scale Monte Carlo simulations on a high performance cluster.
You can find a collection of script to perform simulations on a cluster that runs OpenPBS 
`here <https://github.com/ToFeWe/q-learning-simulation-code>`_.

Furthermore, the  `replication package <https://github.com/ToFeWe/q-learning-replication-code>`_ 
for the aforementioned paper uses this package.


* Free software: MIT license


Installation
------------

TODO


Parameter specification
-----------------------

In the simulation certain (hyper-)parameters have to be defined in a dictionary.
The following keys are necessary:

*  n_agent: Number of agents/firms in the market (*integer*, in paper = 2 or 3)
*  k_memory: Memory length of the agent (*integer*, in paper = 1)
*  discount_rate: Memory length of the agent (*integer*, in paper = 0.95)
*  exploration_rate: Initial probability to take a random action for the epsilon-greedy exploration (*float* between zero and one)
*  beta_decay: Decay parameter for the epsilon-greedy exploration (*float* close to zero)
*  learning_rate: Weight that is given to new information in each round (*float* between zero and one)
*  min_price: Minimal price that agents can set (*integer*, in paper = 0)
*  max_price: Maximal price that agents can set (*integer*, in paper = 5)
*  reservation_price": Reservation price of the consumers (*integer*, in paper = 4)
*  m_consumer: Number of consumers in the market (*integer*, in paper = 60)
*  step: Steps between each price in the set of possible prices (*integer*, in paper = 1)
*  learning_iterations: Maximal number of learning iterations if convergence fails (*integer*, in paper = 1000000000)
*  rounds_convergence: Number of rounds to determine if the agent converged (*integer*, in paper = 100000)
*  Q_star_threshold: Threshold to check if the optimal Q-matrix converged (*float*, in paper = 1e-09)
*  avg_price_rounds: Number of periods to calculate the average price after convergence (*integer*, in paper = 1000)

If you run a Monte Carlo simulation over a parameter grid, you have to define an additional dictionary (PARAMETER_CASES)
with the following keys:
*  beta_max: Maximal beta decay (*float*, in paper = 2e-05)
*  beta_min: Minimal beta decay (*float*, in paper = 1e-08)
*  alpha_max: Maximal learning rate (*float*, in paper = 0.25)
*  alpha_min: Minimal learning rate (*integer*, in paper = 0.025)
*  grid_points: Number of grid points to consider (*integer*, in paper = 100) 
*  path_differ: String that is attached to each file name (*string*)

For the paper, I run 1,000 simulations with different random seeds for each grid point.
The parallelization is done across jobs on the cluster using PBS. 
The See here for an implementation: `PBS-scripts <https://github.com/ToFeWe/q-learning-simulation-code>`_.

Minimal working examples
------------------------

To simulate a single agent you can do the following:

.. code-block:: python
  :linenos:

  from qpricesim.simulations.agents_simulation import train_agents
  
  PARAMETER = {
    "n_agent": 2,
    "k_memory": 1,
    "discount_rate": 0.95,
    "exploration_rate": 1,
    "learning_rate": 0.05,
    "beta_decay": 5e-06,
    "min_price": 0,
    "max_price": 5,
    "reservation_price": 4,
    "m_consumer": 60,
    "step": 1,
    "learning_iterations": 1000000000,
    "rounds_convergence": 100000,
    "Q_star_threshold": 1e-09,
    "avg_price_rounds": 1000
    }

  bool_convergence, state, all_agents = train_agents(parameter=PARAMETER,
                                                     random_seed=45)
    
For running a one iteration in a small grid simulation:

.. code-block:: python
  :linenos:

  from qpricesim.simulations.mc_simulation_job_array import run_single_simulation
  
  PARAMETER_BASE = {
    "n_agent": 2,
    "k_memory": 1,
    "discount_rate": 0.95,
    "exploration_rate": 1,
    "min_price": 0,
    "max_price": 5,
    "reservation_price": 4,
    "m_consumer": 60,
    "step": 1,
    "learning_iterations": 1000000000,
    "rounds_convergence": 100000,
    "Q_star_threshold": 1e-09,
    "avg_price_rounds": 1000
    }

  PARAMETER_CASES = {
    "beta_max": 2e-05,
    "beta_min": 1e-07,
    "alpha_max": 0.25,
    "alpha_min": 0.025,
    "grid_points": 10,
    "path_differ": "2_agents"
  }

  RESULTS = run_single_simulation(
      base_parameter=PARAMETER_BASE,
      cases=PARAMETER_CASES,
      job_array_index=1,
  )    

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
