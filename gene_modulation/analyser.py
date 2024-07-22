# ---------------------------------------------------------
# Plotting to assist in analyzing training results
# ---------------------------------------------------------


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class Analyser:

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.out_path = config.result_folder


    def plot_reward(self):
        ## Reward vs. Episode Plot
        r_cumsum = np.cumsum(np.insert(self.trainer.rs, 0, 0)) 
        r_cumsum = (r_cumsum[50:] - r_cumsum[:-50]) / 50

        # Plot
        plt.plot(r_cumsum)
        plt.title('Reward vs. Episode Plot')
        plt.savefig(os.path.join(self.out_path, 'episode_reward.png'))

        ## Normalized reward vs. Episode Plot
        r_cumsum = np.cumsum(np.insert(self.trainer.normalized_reward, 0, 0)) 
        r_cumsum = (r_cumsum[50:] - r_cumsum[:-50]) / 50

        # Plot
        plt.clf()
        plt.plot(r_cumsum)
        plt.title('Normalized Reward (reward / n_steps to reach target) vs. Episode Plot')
        plt.savefig(os.path.join(self.out_path, 'episode_normalized_reward.png'))
    

    def plot_consecutive_state_visit(self, csv_log_file):
        result_df = pd.read_csv(csv_log_file)
        result_df['consecutive_values'] = (result_df['s1'] != result_df['s1'].shift()).cumsum()

        consecutive_attr = result_df.groupby(['i', 's1', 'consecutive_values'], as_index=False)['consecutive_values'].count()

        consecutive_attr['s1'] = consecutive_attr['s1'].astype('category')
        consecutive_attr['i'] = consecutive_attr['i'].astype('int')

        rel_plot = sns.relplot(
            data=consecutive_attr,
            x='i', y='consecutive_values', col='s1', col_wrap=2
        )
        rel_plot.savefig(os.path.join(self.out_path, 'consecutive_state_visits.png'))