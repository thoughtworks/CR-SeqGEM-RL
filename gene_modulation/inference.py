
## Imports
import os
import pickle
import numpy as np
from pyboolnet import attractors

from .config import Config


class Inference:
    def __init__(self, env, boolean_model, config:Config, learnt_q_table, logger=None):

        self.env = env

        self.boolean_model = boolean_model
        self.update_scheme = config.update_scheme
        self.target_state_str = config.target_state_str
        self.attr_list, self.agent_initial_state_list = self.__get_initial_attractor_states()

        self.learnt_q_table = learnt_q_table
        self.learnt_q_table_attr = self.__process_learnt_q_table()

        self.logger=logger
        self.result_path = config.result_folder
        self.output_file = config.output_file
    

    ## Create a list of initial states based on the attractors of the Boolean network
    def __get_initial_attractor_states(self):
        
        # Get the list of attractors
        attr = attractors.compute_attractors(self.boolean_model, self.update_scheme)

        attr_list = []
        for i in range(len(attr['attractors'])):
            attr_list.append(attr['attractors'][i]['state']['str'])

        attr_list = tuple(attr_list)

        # Use the attractor states except the target attractor state as initial states
        attr_list_wo_target_attr = list(attr_list)
        attr_list_wo_target_attr.remove(self.target_state_str)  # Remove the target attractor state from attr_list
        agent_initial_state_list = attr_list_wo_target_attr    # Assign the remaining attr_list to agent_initial_state_list

        return attr_list, agent_initial_state_list


    ## Get the max value and the column (i.e. the gene) that has the max value for a row (i.e. state) in the learnt Q-table. 
    # This gene will be perturbed if the agent is in that state.
    def __process_learnt_q_table(self):

        # Get max value corresponding to each row (i.e. state) in the Q-table
        self.learnt_q_table['max_val'] = self.learnt_q_table.max(axis=1)

        # Get the column no. (i.e. the gene) that has max value for a row
        self.learnt_q_table['max_val_idx'] = self.learnt_q_table.values.argmax(axis=1)

        # TODO FOR LATER: Check if there are more than one column with max value

        # Filter the Q-table with data for attractor states only
        learnt_q_table_attr = self.learnt_q_table[self.learnt_q_table.index.isin(self.attr_list)].copy(deep=True)

        return learnt_q_table_attr
        

    def save_q_table_with_maxval(self, file_name_suffix='with_maxval_forInference', as_df=False):
        with open(os.path.join(self.result_path, f'{self.output_file}_'f'{file_name_suffix}.pkl'), 'wb') as f:
            pickle.dump(self.learnt_q_table, f)

        if as_df:
            self.learnt_q_table.to_csv(os.path.join(self.result_path, f'{self.output_file}_'f'{file_name_suffix}.pkl'), 
                                       index_label='State')


    def infer(self):

        # Initialize additional variables
        actions = np.arange(0, len(self.boolean_model), dtype=int)
        result_dict = {}

        for state in self.agent_initial_state_list:
            agent_initial_state = np.array(list(state), dtype=int)

            self.env.reset()
            # Set the env initial state
            self.env.agent_initial_state = agent_initial_state
            

            i = 0
            t = 0
            terminated = False
            truncated = False

            self.env.agent_state = self.env.agent_initial_state
            s_ori = self.env.agent_state
            s_arr = s_ori.copy()
            
            result_dict[i] = {}

            while (not terminated) & (not truncated):
            
                s_arr = s_ori.copy()

                s = ''.join(map(str, s_arr))
                a = self.learnt_q_table_attr[self.learnt_q_table_attr.index == s]['max_val_idx'].values               
                s1, _, terminated, truncated, info_dict = self.env.step(a)

                if self.logger is not None:
                    self.logger.info(f"\t {state}\t {i}\t {t}\t {s}\t {a}\t \
                    {''.join(map(str, info_dict['agent_perturbed_state']))}\t \
                    {''.join(map(str, s1))}")

                    
                result_dict[i][t] = {'s': s, 'a': a, 's_perturbed': ''.join(map(str, info_dict['agent_perturbed_state'])), 
                                    's1': ''.join(map(str, s1))}

                s_ori = s1
                s_arr = s_ori.copy()

                t += 1
            i += 1
