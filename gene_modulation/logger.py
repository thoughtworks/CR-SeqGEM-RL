# ---------------------------------------------------------
# Logger related imports and settings
# ---------------------------------------------------------


import os
import logging
from datetime import datetime
import pandas as pd

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


class Logger:
    
    @staticmethod
    def get_logger(name, log_folder, output_file, mode='general'):
        if mode == 'general':
            file_suffix = 'general_log_file'
        elif mode == 'train':
            file_suffix = 'training_result'
        elif mode == 'infer':
            file_suffix = 'inferencing_result'
        else:
            raise Exception('Undefined logging mode.')
        file_name = datetime.now().strftime(os.path.join(log_folder,
                    f'{output_file}_{file_suffix}_%d_%m_%Y__%H_%M_%S.log'))
        
        return Logger.__setup_logger(name, file_name)
    

    @staticmethod
    def __setup_logger(name, log_file, level=logging.INFO):
        ## To setup required loggers

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger
    

    ## Save model training results
    @staticmethod
    def save_training_logger(logger, file_name):
        handler = logger.handlers[0]
        filename = handler.baseFilename
        col_names = ['info', 'i', 't', 's', 'a', 's_perturbed', 's1', 'Q[s,a]', 'r_sum_i']
        result_df = pd.read_csv(filename, sep='\t\s*', names=col_names, engine='python', dtype='object')
        result_df.drop(columns='info', inplace=True)
        result_df.to_csv(file_name, index=False)


    ## Save model inferencing results
    @staticmethod
    def save_inferencing_logger(logger, file_name):
        handler = logger.handlers[0]
        filename = handler.baseFilename
        col_names = ['info', 'Initial_State', 'i', 't', 's', 'a', 's_perturbed', 's1']
        inference_df = pd.read_csv(filename, sep='\t\s*', names=col_names, engine='python', dtype='object')
        inference_df.drop(columns='info', inplace=True)
        inference_df.to_csv(file_name, index=False)