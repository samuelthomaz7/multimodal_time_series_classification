
import pickle
import numpy as np
import os
import pandas as pd
import torch 
import random

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_all_results():
    all_dirs = os.listdir('./model_checkpoints')

    info = []
    for directory in all_dirs:
        execution_info = {
            'directory': directory,
            'model_name': directory.split('_')[0],
            'dataset': directory.split('_')[1],
            'seed': directory.split('_')[2]
        }

        info.append(execution_info)

        if 'model_history.pkl' in os.listdir('./model_checkpoints/' + directory):
            with open('./model_checkpoints/' + directory + '/model_history.pkl', 'rb') as f:
                history = pickle.load(f) 

            execution_info['max_accuracy'] = max(history.history['val_accuracy'])
            execution_info['min_val_loss'] = min(history.history['val_loss'])
            execution_info['epochs'] = len(history.history['val_loss'])
        else:
            execution_info['max_accuracy'] = None
            execution_info['min_val_loss'] = None
            execution_info['epochs'] = None

    complete_data = pd.DataFrame(info)
        
    return complete_data