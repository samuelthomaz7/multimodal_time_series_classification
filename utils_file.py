
import pickle
import numpy as np
import os
import pandas as pd
import torch 
import random

from tqdm import tqdm

from input.reading_datasets import read_dataset_from_file
from preprocessing.get_dummies_labels import GetDummiesLabels
from preprocessing.train_test_split_module import TrainTestSplit

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


def training_nn_for_seeds(used_model, datasets = [], seeds = []):
    for dataset in tqdm(datasets):
        for random_state in tqdm(seeds):
            print(f'{dataset} - {random_state}')
            used_dataset = read_dataset_from_file(dataset_name = dataset)
            X, y, metadata = used_dataset

            get_dummies_object = GetDummiesLabels(
                X_raw= X,
                y_raw= y,
                metadata= metadata
            )

            X, y = get_dummies_object.transform()

            train_test_object = TrainTestSplit(
                X_raw= X,
                y_raw= y,
                metadata= metadata,
                random_state = random_state
            )

            X_train, X_test, y_train, y_test = train_test_object.transform()

            model = used_model(
                X_train=X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test,
                metadata = metadata,
                random_state = random_state
            )

            if len(os.listdir('./model_checkpoints/' + model.model_folder)) != 0 :
                pass
            else:
                model.training_process()