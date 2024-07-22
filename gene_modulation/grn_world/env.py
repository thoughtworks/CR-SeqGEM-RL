# ---------------------------------------------------------
# Define coupled model environment
# ---------------------------------------------------------


## Imports
import numpy as np
from typing import Any
import gymnasium as gym
from gymnasium import spaces
from pyboolnet import state_transition_graphs as STGs
from pyboolnet import attractors
from scipy.spatial.distance import hamming


class GRNEnv(gym.Env):
    def __init__(self, boolean_model, update_scheme, agent_initial_state:Any=None, target_state:Any=None):    

        self.boolean_model = boolean_model
        self.update_scheme = update_scheme
        self.agent_initial_state = agent_initial_state
        self.agent_state = agent_initial_state
        self.target_state = target_state
        
        self.terminated_condition = False
        self.truncated_condition = False


        self.observation_space = spaces.MultiBinary(agent_initial_state.shape[0])

        # Actions correspond to the genes whose state can be modulated. This is equal to the length of the vector 
        # defining an agent's state   
        self.action_space = spaces.Discrete(agent_initial_state.shape[0])
        self.actions = np.arange(0, len(boolean_model), dtype=int)


    ## Method: Compute the Hamming distance between current agent state and the target state
    def _get_info(self):
        return hamming(self.agent_state, self.target_state) * len(self.agent_state)


    ## Method: Flip the bit at the specified location of the vector
    def gene_perturb(self, action):
        self.agent_state[action] = self.agent_state[action]^1
        return self.agent_state


    ## Method: Find the attractor(s) corresponding to a perturbed state
    def find_attractor(self):
        agent_state_str = ''.join(map(str, self.agent_state.tolist()))

        steady = []
        cyclic = []

        stg = STGs.primes2stg(self.boolean_model, self.update_scheme, agent_state_str)        
        steady, cyclic = attractors.compute_attractors_tarjan(stg)
    
        ## TODO FOR LATER: Currently, the first steady state attractor is used. More than one steady 
        # state attractor is not handled yet. Cyclic attractors, if any, are also not handled yet.

        if steady:
            self.agent_state = np.array(list(steady[0]), dtype=int)
        
        return self.agent_state


    ## REWARD FUNCTION:
    def reward_func(self):
        info = self._get_info()

        if info != 0:
            reward = 1 / info
        else:           # Give higher reward if the agent reaches the target
            reward = 2

        return reward


    ## Method: reset
    def reset(self, *, seed=None, options=None):
        # Following line is needed to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.agent_state = self.np_random.integers(0, 1, size=self.agent_initial_state.shape[0], dtype=int, endpoint=True)

        observation = self.agent_state
        info = self._get_info()

        return observation, {'hamming': info}


    ## Method: step
    def step(self, action):
        self.agent_state = self.gene_perturb(action)
        agent_perturbed_state = self.agent_state.copy()        
        self.agent_state = self.find_attractor()

        ## An episode is completed if the agent has reached the target
        # Hamming distance is used as a measure of the distance between agent current state and the target state
        # If the Hamming distance = 0, the agent current state is equal to the target state
        self.terminated_condition = self._get_info() == 0.0

        reward = self.reward_func()
        observation = self.agent_state

        info = self._get_info()
        info_dict = {'agent_perturbed_state': agent_perturbed_state, 'info': info}

        return observation, reward, self.terminated_condition, self.truncated_condition, info_dict


    def close(self):
            pass


    def render(self, mode=None):
        print(self.log)
        self.log = ''