
## Imports
import os
import pickle
import numpy as np
import pandas as pd
from itertools import product

from .config import Config


class Trainer:
    def __init__(self, env, boolean_model, config:Config,
                 logger=None, env_reset_seed=None):

        self.env = env
        self.env_reset_seed = env_reset_seed

        self.boolean_model = boolean_model       
        self.invariant_genes = config.invariant_genes

        self.alpha = config.alpha
        self.gamma = config.gamma
        self.num_episodes = config.num_episodes
        self.rs = np.zeros([self.num_episodes])
        self.n_steps_s = np.zeros([self.num_episodes])
        self.Q = self.__initialize_q_table()

        self.logger=logger
        self.result_path = config.result_folder
        self.output_file = config.output_file
    

    def __initialize_q_table(self):
        q_s_list = list(map(list, product([0, 1], repeat=len(self.boolean_model)))) # type: ignore

        Q = {}
        for state in q_s_list:
            state = ''.join(map(str, state)) # type: ignore
            Q[state] = np.zeros(len(state))
        return Q
    

    @property
    def normalized_reward(self):
        return self.rs / self.n_steps_s
    

    def train(self):

        result_dict = {}

        ## Training
        for i in range(self.num_episodes):
            r_sum_i = 0
            t = 0
            n_steps = 0
            terminated = False
            truncated = False

            s_ori = self.env.reset(seed=self.env_reset_seed)[0]
            s_arr = s_ori.copy()

            result_dict[i] = {}

            while (not terminated) & (not truncated):
                s_arr = s_ori.copy()
                s = ''.join(map(str, s_arr))
                
                
                ## Select action (i.e. gene for modulation) among the genes that are allowed (i.e. exclude the invariant_genes)
                # If invariant_genes is specified, then exclude those genes from the list of possible actions
                if self.invariant_genes:
                    a_tmp = self.Q[s] + np.random.randn(1, self.env.action_space.n)*(1./(i/10+1))
                    
                    idx_sorted = np.argsort(-a_tmp)
                            
                    # Identify indices corresponding to invariant_genes that need to be deleted from the list of possible actions
                    del_idx=np.flatnonzero(np.isin(idx_sorted, self.invariant_genes))

                    # Get a truncated and sorted list of indices
                    idx_sorted_truncated = np.delete(idx_sorted, del_idx)
                    
                    # The first element refers to the index (action) with highest value
                    a = idx_sorted_truncated[0]

                else:
                    a = np.argmax(self.Q[s] + np.random.randn(1, self.env.action_space.n)*(1./(i/10+1)))


                # Perform the identified action
                s1, r, terminated, truncated, info_dict = self.env.step(a)
                s1_str = ''.join(map(str, s1))

                # Update the Q-table based on the action performed and the reward received
                self.Q[s][a] = (1 - self.alpha)*self.Q[s][a] + self.alpha*(r + self.gamma*np.max(self.Q[s1_str]))

                # Add received reward to total episode reward
                r_sum_i += r*self.gamma**t

                if self.logger is not None:
                    self.logger.info(f"\t {i}\t {t}\t {s}\t {a}\t \
                    {''.join(map(str, info_dict['agent_perturbed_state']))}\t \
                    {s1_str}\t {self.Q[s][a]}\t {r_sum_i}")

                        
                result_dict[i][t] = {'s': s, 'a': a, 's_perturbed': ''.join(map(str, info_dict['agent_perturbed_state'])), 
                                's1': s1_str, 'Q[s,a]': self.Q[s][a], 'r_sum_i': r_sum_i}


                # Update the states
                s_ori = s1
                s_arr = s_ori.copy()

                # Make required increments
                n_steps += 1
                t += 1

            # Record the total episode reward
            self.rs[i] = r_sum_i
            self.n_steps_s[i] = n_steps
    
    
    def save_q_table(self, file_name='Learnt_Q_Table', as_df=False):
        with open(os.path.join(self.result_path, f'{file_name}.pkl'), 'wb') as f:
            pickle.dump(self.Q, f)

        if as_df:
            Q_df = pd.DataFrame.from_dict(self.Q, orient='index', columns=list(self.boolean_model.keys()))
            Q_df.to_csv(os.path.join(self.result_path, f'{self.output_file}_{file_name}.csv'), index_label='State')