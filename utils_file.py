
import pickle
import numpy as np
import os
import pandas as pd
import torch 
from torch import nn
import random
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from modality_info import modalities

from input.reading_datasets import read_dataset_from_file
from input.time_series_module import TimeSeriesDataset
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


def training_nn_for_seeds(used_model, device = 'cuda', datasets = [], seeds = [], is_multimodal = False, is_debbug = False, num_ensembles = None):
    for dataset in tqdm(datasets):
        model_count = 1
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
            X_train, X_test, y_train, y_test = torch.from_numpy(X_train).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(y_train).to(device), torch.from_numpy(y_test).to(device)


            train_dataset = TimeSeriesDataset(
                data=X_train,
                labels=y_train,
                metadata=metadata
            )

            test_dataset = TimeSeriesDataset(
                data=X_test,
                labels=y_test,
                metadata=metadata
            )

            if num_ensembles == None:

                model = used_model(
                    train_dataset = train_dataset,
                    test_dataset = test_dataset,
                    metadata = metadata,
                    random_state = random_state,
                    dataset_name = dataset,
                    device = device
                ).to(device)

                if (len(os.listdir('./model_checkpoints/' + model.model_folder)) != 0) and (is_debbug == False):
                    pass
                else:
                    model.fit()

            
            else:
                activation_function = nn.Softmax(dim = 1) if train_dataset.labels.shape[1] != 2 else nn.Sigmoid()
                best_models = []
                
                for model_num in range(1, num_ensembles+1):

                    model = used_model(
                        train_dataset = train_dataset,
                        test_dataset = test_dataset,
                        metadata = metadata,
                        random_state = random_state,
                        dataset_name = dataset,
                        device = device,
                        model_num = model_count
                    ).to(device)

                    if (len(os.listdir('./model_checkpoints/' + model.model_folder)) != 0) and (is_debbug == False) :
                        pass
                    else:
                        model.fit()
                    
                    model_count += 1

                    load_model = torch.load(f = './model_checkpoints/' + model.model_folder + '/best_model.pth') 
                    model.load_state_dict(load_model)

                    best_models.append(model)



                if ('metrics.pkl' in os.listdir('./model_checkpoints/' + model.model_folder)) or (is_debbug == False) :
                    pass
                else:

                    predictions = []
                    train_predictions = []
                    
                    for model in best_models:

                        with torch.inference_mode():   
                            logits = model(test_dataset.data.type(torch.float32))
                            predictions.append(activation_function(logits))

                            train_logits = model(train_dataset.data.type(torch.float32))
                            train_predictions.append(activation_function(train_logits))
                        
                        stacked_tensors = torch.stack(predictions)
                        ensembled_predictions = torch.mean(stacked_tensors, dim = 0)

                        stacked_tensors_train = torch.stack(train_predictions)
                        ensembled_predictions_train = torch.mean(stacked_tensors_train, dim = 0)
                    
                    accuracy = accuracy_score(y_true = torch.argmax(test_dataset.labels, dim = 1).cpu(), y_pred = torch.argmax(ensembled_predictions, dim = 1).cpu())
                    train_accuracy = accuracy_score(y_true = torch.argmax(train_dataset.labels, dim = 1).cpu(), y_pred = torch.argmax(ensembled_predictions_train, dim = 1).cpu())

                    
                    history = {
                        'train_loss' : [0], 
                        'test_loss': [0], 
                        'train_accuracy': [train_accuracy], 
                        'test_accuracy': [accuracy], 
                        'epochs': [9999]

                    }

                    metrics = {
                        'history': history,
                        'traning_time': 99999
                    }

                    with open('./model_checkpoints/' + model.model_folder.split('/')[0] + '/metrics.pkl', 'wb') as f:  # open a text file
                        pickle.dump(metrics, f)
                
                model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                if 'params_info.csv' in os.listdir('.'):
                    params_info = pd.read_csv('params_info.csv')

                    if params_info.loc[(params_info.model_name == model.model_name) & (params_info.dataset_name == model.metadata['problemname'])].shape[0] != 0:
                        pass
                    else:
                        params_info_new = pd.DataFrame({
                            'model_name': [model.model_name],
                            'dataset_name': [model.metadata['problemname']],
                            'model_params': model_params
                        })

                        pd.concat([params_info, params_info_new], axis = 0).to_csv('params_info.csv', index=False)


                else:
                    params_info = pd.DataFrame({
                        'model_name': [model.model_name],
                        'dataset_name': [model.metadata['problemname']],
                        'model_params': model_params
                    })

                    params_info.to_csv('params_info.csv', index=False)
                    


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def get_all_results(grouped = False):


    all_dirs = os.listdir('./model_checkpoints')

    info = []
    for directory in all_dirs:
        execution_info = {
            'directory': directory,
            'model_name': directory.split('_')[0],
            'dataset': directory.split('_')[1],
            'seed': int(directory.split('_')[2])
        }


        

        if 'metrics.pkl' in os.listdir('./model_checkpoints/' + directory):
            with open('./model_checkpoints/' + directory + '/metrics.pkl', 'rb') as f:
                history = pickle.load(f) 

            execution_info['max_train_accuracy'] = max(history['history']['train_accuracy'])
            execution_info['max_test_accuracy'] = max(history['history']['test_accuracy'])
            execution_info['epochs'] = len(history['history']['epochs'])
            execution_info['execution_time'] = (history['traning_time'])
            execution_info['time_per_epoch'] = history['traning_time']/execution_info['epochs']


        info.append(execution_info)
        complete_data = pd.DataFrame(info).sort_values(by = ['dataset', 'model_name','seed']).reset_index(drop = True)
    
    if grouped:

        agg_data =  complete_data.groupby(['dataset', 'model_name']).agg({
            'max_train_accuracy': 'mean',
            'max_test_accuracy': 'mean',
            'epochs': 'mean',
            'execution_time': 'mean',
            'time_per_epoch': 'mean'
        }).reset_index()

        return agg_data

    else:
        return complete_data