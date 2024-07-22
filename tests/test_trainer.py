
## Imports
import os
import numpy as np
import pandas as pd
from gymnasium.wrappers.time_limit import TimeLimit
from kneed import KneeLocator

from .config import Config
from gene_modulation.utils import read_boolean_model
from gene_modulation.grn_world.env import GRNEnv
from gene_modulation.trainer import Trainer
from gene_modulation.logger import Logger


## Get the required inputs from config
config = Config()


def test_trainer():

    # Initialization of Log files
    test_training_result = Logger.get_logger('test_training_logger', config.logging_folder, config.output_file, mode='train')


    # Initialization of Boolean model
    boolean_model = read_boolean_model(os.path.join(config.artefacts_folder, config.boolean_model_file))

    # Initial and target states of the Boolean network
    agent_initial_state = np.array(list(config.agent_initial_state_str), dtype=int)
    target_state = np.array(list(config.target_state_str), dtype=int)


    # Initialize the env
    env = GRNEnv(boolean_model=boolean_model, update_scheme=config.update_scheme, 
                 agent_initial_state=agent_initial_state, target_state=target_state)
    env = TimeLimit(env, max_episode_steps=config.max_episode_steps)


    ## Model training
    trainer = Trainer(env, boolean_model, config, logger=test_training_result, env_reset_seed=1234)
    trainer.train()

    # trainer.save_q_table(as_df=True)

    test_csv_log_file = os.path.join(config.result_folder, f'test_{config.output_file}_ModelTraining.csv')
    Logger.save_training_logger(test_training_result, test_csv_log_file)


    ## Use model convergence as a criterion to test model training. This is achieved using kneed library 
    # (https://pypi.org/project/kneed/#find-knee) to find the knee/elbow point during model training. If 
    # any such point is found, then the test passes.
    test_trainer_output = pd.read_csv(os.path.join(config.result_folder, f'test_{config.output_file}_ModelTraining.csv'), 
                                      dtype='object')

    # Change datatypes
    test_trainer_output[['i', 't']] = test_trainer_output[['i', 't']].astype('int')
    test_trainer_output['r_sum_i'] = test_trainer_output['r_sum_i'].astype('float')

    # Group on episodes ('i') and sum the rewards received in each episode
    test_trainer_output_grouped = test_trainer_output.groupby('i', as_index=False).sum()


    # The argument 'online' is marked as True indicating that kneed runs in online mode and “corrects” itself 
    # by continuing to overwrite any previously identified knees. This allows for detecting 'global' knee point.
    kneedle = KneeLocator(test_trainer_output_grouped['i'], test_trainer_output_grouped['r_sum_i'], 
                          S=1.0, curve='convex', direction='decreasing', online=True)
    
    if kneedle.knee:
        assert True