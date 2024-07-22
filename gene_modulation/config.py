# ---------------------------------------------------------
# Set inputs and hyperparameters
# ---------------------------------------------------------


from dataclasses import dataclass
import os


@dataclass
class Config:

    data_folder = './data'    
    logging_folder = './logs'
    result_folder = './results'

    ## Create directories for storing logs and results if they do not exist
    if not os.path.exists(logging_folder):
        os.makedirs(logging_folder)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    
    ## Boolean model related inputs
    boolean_model_file = 'Cardiac_development.txt'
    output_file = 'cardiac'  # This is used as prefix for the output filenames
    update_scheme = 'asynchronous'  # Boolean model simulation update scheme

    ## Initial state of the Boolean network
    agent_initial_state_str = '000000000000000'

    # Initial state for inference
    agent_initial_allzeros_state_str = '000000000000000'


    ## Target state of the Boolean network
    # Cardiac development network
    target_state_str = '100010010101100'  # FHF state


    ## Specify the invariant genes i.e. the genes that the model should not modulate
    invariant_genes = ()


    ## Q-learning model: Hyperparameters
    alpha = 0.1     # Learning rate
    gamma = 0.95    # Discount factor
    num_episodes = 300
    max_episode_steps = 100
