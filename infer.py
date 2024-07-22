
## Imports
import os
import numpy as np
import pandas as pd

from gene_modulation.config import Config
from gene_modulation.logger import Logger
from gene_modulation.utils import read_boolean_model
from gene_modulation.grn_world.env import GRNEnv
from gene_modulation.inference import Inference


## Get the required inputs from config
config = Config()

# Initialize the Log files
logger = Logger.get_logger('logger', config.logging_folder, config.output_file, mode='general')
inferencing_result = Logger.get_logger('inferencing_logger', config.logging_folder, config.output_file, mode='infer')

# Initialize the Boolean model
boolean_model = read_boolean_model(os.path.join(config.data_folder, config.boolean_model_file))
logger.info(f'Boolean model: \n{boolean_model}')

# Initial state of the Boolean network
# For inferencing, the initial states would be all the attractor states except the target attractor. 
# They would be used one at a time to find the modulations required for transitioning them to the target attractor.
# This list of initial states is created based on the attractors of the Boolean network.

# Target state of the Boolean network
target_state = np.array(list(config.target_state_str), dtype=int)


# Learnt Q-Table: Read from a saved file
learnt_q_table = pd.read_csv(os.path.join(config.result_folder, f'{config.output_file}_Learnt_Q_Table.csv'), 
                             dtype={'State': object})
learnt_q_table.set_index(learnt_q_table['State'], inplace=True)
learnt_q_table.drop(columns='State', inplace=True)


## Initialize the env with an allzeros agent state
agent_initial_allzeros_state = np.array(list(config.agent_initial_allzeros_state_str), dtype=int)

env = GRNEnv(boolean_model=boolean_model, 
             update_scheme=config.update_scheme, 
             agent_initial_state=agent_initial_allzeros_state, 
             target_state=target_state)     ## TODO FOR LATER: Add time limit to truncate an episode


## Inferencing
# Perform inferencing i.e. find out the modulations required for transitioning the initial attractor states to target state
inference = Inference(env, boolean_model, config, learnt_q_table, logger=inferencing_result)
inference.infer()

# Save the Q-table along with maxval indicating the gene to be modulated at each Boolean network state
inference.save_q_table_with_maxval(as_df=True)

# Save the inferencing results derived from the log file
csv_log_file = os.path.join(config.result_folder, f'{config.output_file}_ModelInferencing.csv')
Logger.save_inferencing_logger(inferencing_result, csv_log_file)