
## Imports
import os
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit

from gene_modulation.config import Config
from gene_modulation.logger import Logger
from gene_modulation.utils import read_boolean_model
from gene_modulation.grn_world.env import GRNEnv
from gene_modulation.trainer import Trainer
from gene_modulation.analyser import Analyser


## Get the required inputs from config
config = Config()

# Initialize the Log files
logger = Logger.get_logger('logger', config.logging_folder, config.output_file, mode='general')
training_result = Logger.get_logger('training_logger', config.logging_folder, config.output_file, mode='train')


# Initialize the Boolean model
boolean_model = read_boolean_model(os.path.join(config.data_folder, config.boolean_model_file))

# Initial and target states of the Boolean network
agent_initial_state = np.array(list(config.agent_initial_state_str), dtype=int)
target_state = np.array(list(config.target_state_str), dtype=int)


logger.info(f'Boolean model: \n{boolean_model}')
logger.info(f'Initial state and target state shapes: {agent_initial_state.shape} \t {target_state.shape}')
logger.info(f'Initial and target states: {agent_initial_state} \t {target_state}')


# Initialize the env
env = GRNEnv(boolean_model=boolean_model, update_scheme=config.update_scheme, 
             agent_initial_state=agent_initial_state, target_state=target_state)
env = TimeLimit(env, max_episode_steps=config.max_episode_steps)

logger.info(f'env reset: {env.reset()}')
logger.info(f'Observation space: {env.observation_space}')
logger.info(f'Observation space sample: {env.observation_space.sample()}')
logger.info(f'Action space: {env.action_space}')
logger.info(f'Action space sample: {env.action_space.sample()}')
logger.info(f'Perform a step: {env.step(2)}')


## Model training
trainer = Trainer(env, boolean_model, config, logger=training_result, env_reset_seed=None)
trainer.train()

trainer.save_q_table(as_df=True)

csv_log_file = os.path.join(config.result_folder, f'{config.output_file}_ModelTraining.csv')
Logger.save_training_logger(training_result, csv_log_file)

analyzer = Analyser(trainer, config)
analyzer.plot_reward()
analyzer.plot_consecutive_state_visit(csv_log_file)