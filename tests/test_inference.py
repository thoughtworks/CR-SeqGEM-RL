
## Imports
import os
import filecmp
import numpy as np
import pandas as pd

from .config import Config
from gene_modulation.utils import read_boolean_model
from gene_modulation.grn_world.env import GRNEnv
from gene_modulation.inference import Inference
from gene_modulation.logger import Logger


## Get the required inputs from config
config = Config()


def test_inference():

    # Initialize the Log files
    test_inferencing_result = Logger.get_logger('test_inferencing_logger', config.logging_folder, config.output_file, mode='infer')

    # Initialize the Boolean model
    boolean_model = read_boolean_model(os.path.join(config.artefacts_folder, config.boolean_model_file))

    # Initial state of the Boolean network
    # For inferencing, the initial states would be all the attractor states except the target attractor. 
    # They would be used one at a time to find the modulations required for transitioning them to the target attractor.
    # This list of initial states is created based on the attractors of the Boolean network.


    # Target state of the Boolean network
    target_state = np.array(list(config.target_state_str), dtype=int)


    # Learnt Q-Table: Read from a saved file
    learnt_q_table = pd.read_csv(os.path.join(config.artefacts_folder, config.inference_learnt_Q_table_artefact), 
                                dtype={'State': object})
    learnt_q_table.set_index(learnt_q_table['State'], inplace=True)
    learnt_q_table.drop(columns='State', inplace=True)


    ## Initialize the env with an allzeros agent state
    agent_initial_allzeros_state = np.array(list(config.agent_initial_allzeros_state_str), dtype=int)
    env = GRNEnv(boolean_model=boolean_model, update_scheme=config.update_scheme, 
                 agent_initial_state=agent_initial_allzeros_state, target_state=target_state)


    ## Inferencing
    # Perform inferencing i.e. find out the modulations required for transitioning the initial attractor state to target state
    inference = Inference(env, boolean_model, config, learnt_q_table, logger=test_inferencing_result)
    inference.infer()

    # Save the inferencing results derived from the log file
    test_csv_log_file = os.path.join(config.result_folder, f'test_{config.output_file}_ModelInferencing.csv')
    Logger.save_inferencing_logger(test_inferencing_result, test_csv_log_file)


    artefact_data = os.path.join(config.artefacts_folder, config.inference_output_artefact)
    test_inference_output = os.path.join(config.result_folder, f'test_{config.output_file}_ModelInferencing.csv')
    assert filecmp.cmp(artefact_data, test_inference_output) == True