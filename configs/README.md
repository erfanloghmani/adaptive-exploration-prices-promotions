# Config Files

This directory contains the config files for the experiments in the paper. Each config file is a `json` file contatining the information about the experiment 

## File structure

- `algs`: contains a list of algorithms with specific parameters. Each algorithm is a dictionary with keys:
    - `name`: String value that represents the identifier of the method with specific settings in the outputs.
    - `alg_module`: The module that contains the algorithm class.
    - `alg_class`: Name of the class
    - `env`: Whether the environment is contextual or not. `NonContextual` or `Contextual`
    - `active`: Boolean on whether this algorithm setting is included or not. You could set this to `false` if you don't want a specific parameter to run.
    - `params`: Dictionary containing the inputs to the algorithm constructor. Look at the documentation string of each algorithm for details.

- `globals`: A dictionary of other settings shared between the algorithms.
    - `exp_name`: The name of the experiment used for saving the results
    - `negative_beta_option`: How to deal with the negative betas. Either `max` use the maximum price if beta is negative, or `zero`.
    - `show_plays`: Boolean determining whether to log to interaction (prices, promotions, choice) at each time step or not.
    - `write_runs`: Boolean determining whether to save the results of the runs in checkpoints before the run is completed.
    - `checkpoint`: Number of steps between checkpoints
    - `costs`: What marginal costs to use, `zero`, `nielsen`, or `random`.
    - `K`: Number of products
    - `max_T`: Time horizon of the experiment
    - `log_prog_every`: Time intervals between logging what step each algorithm is at.
    - `repeat`: Number of repeats for each algorithm.
    - `tau`: Number of initial exploration rounds.
    - `el`: Lower bound for the prices
    - `u`: Upper bound for the prices
    - If the environment is non-contextual it will require:
        - `dataset`: For non-contextual environments, what set of parameters to use.
    - And if it is contextual it will require:
        - `context_size`: Dimensionality of the context vector
        - `model_class`: The parameter model class that maps contexts to parameters.
        - `model_file`: Path to the file containing parameters of the model.
        - `model_init_params`: Dictionary of the inputs for the constructor of the model.