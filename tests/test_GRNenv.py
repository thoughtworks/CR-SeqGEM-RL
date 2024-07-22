
## Imports
import os
import numpy as np
from pyboolnet.file_exchange import bnet2primes
import json

from gene_modulation.grn_world.env import GRNEnv
from .config import Config
from gene_modulation.utils import read_boolean_model


def test_GRNEnv():
    ## Get the required inputs from config
    config = Config()

    # Initialize the Boolean model
    boolean_model = read_boolean_model(os.path.join(config.artefacts_folder, config.boolean_model_file))
    print(f'Boolean model: {boolean_model}')

    # Initial and target states of the Boolean network
    agent_initial_state = np.array(list(config.agent_initial_state_str), dtype=int)
    target_state = np.array(list(config.target_state_str), dtype=int)


    # Initialize the env
    env = GRNEnv(boolean_model=boolean_model, update_scheme=config.update_scheme, 
                 agent_initial_state=agent_initial_state, target_state=target_state)

    s1, reward, _, _, info_dict = env.step(11)

    out_dict = {'s': config.agent_initial_state_str,
                's1': ''.join(map(str, s1)), 
                'reward': reward, 
                'agent_perturbed_state': ''.join(map(str, info_dict['agent_perturbed_state']))}     # Convert np arrays to strings for writing to json


    ## One-time activity to store the correct expected result of this run to an artefact file
    if not os.path.exists(os.path.join(config.artefacts_folder, config.GRNenv_output_artefact)):
        with open(os.path.join(config.artefacts_folder, config.GRNenv_output_artefact), 'w') as f:
            json.dump(out_dict, f)


    # Read the file containing expected result
    with open(os.path.join(config.artefacts_folder, config.GRNenv_output_artefact)) as f:
        artefact_data = json.load(f)


    assert out_dict == artefact_data
