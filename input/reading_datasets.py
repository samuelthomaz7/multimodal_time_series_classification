import os
import pickle
import pandas as pd
from aeon import datasets
from tqdm import tqdm

from preprocessing.get_dummies_labels import GetDummiesLabels
from preprocessing.train_test_split_module import TrainTestSplit

data = {
    "Dataset": [
        "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions",
        "CharacterTrajectories", "Cricket", "DuckDuckGeese", "EigenWorms",
        "Epilepsy", "EthanolConcentration", "ERing", "FaceDetection",
        "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat",
        "JapaneseVowels", "Libras", "LSST", "InsectWingbeat", "MotorImagery",
        "NATOPS", "PenDigits", "PEMS-SF", "Phoneme", "RacketSports",
        "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
        "StandWalkJump", "UWaveGestureLibrary"
    ],
    "Train Cases": [
        275, 15, 40, 1422, 108, 60, 128, 137, 261, 30, 5890, 316, 320, 150, 204,
        270, 180, 2459, 30000, 278, 180, 7494, 267, 3315, 151, 268, 200, 6599,
        12, 120
    ],
    "Test Cases": [
        300, 15, 40, 1436, 72, 40, 131, 138, 263, 30, 3524, 100, 147, 850, 205,
        370, 180, 2466, 20000, 100, 180, 3498, 173, 3353, 152, 293, 180, 2199,
        15, 320
    ],
    "Dimensions": [
        9, 2, 6, 3, 6, 1345, 6, 3, 3, 4, 144, 28, 10, 3, 61, 12, 2, 6, 200, 64,
        24, 2, 963, 11, 6, 2, 7, 13, 4, 3
    ],
    "Length": [
        144, 640, 100, 182, 1197, 270, 17984, 206, 1751, 65, 62, 50, 400, 152, 
        405, 29, 45, 36, 78, 3000, 51, 8, 1447, 217, 30, 896, 1152, 93, 2500, 315
    ],
    "Classes": [
        25, 3, 4, 20, 12, 5, 5, 4, 4, 6, 2, 2, 4, 26, 2, 9, 15, 14, 10, 2, 6, 10,
        7, 39, 4, 2, 2, 10, 4, 8
    ]
}

available_seires = pd.DataFrame(data)

def read_dataset_from_file(dataset_name):
    with open(f'./downloaded_datasets/{dataset_name}.pkl', 'rb') as f:
            dataset = pickle.load(f) 


    return dataset

def get_all_datasets(read_from_path = True):

    if read_from_path == True:
        print('Reading from path')
        with open('./downloaded_datasets/datasets_info.pkl', 'rb') as f:
            datasets_info = pickle.load(f) 

        return datasets_info
    
    else:
        print('Downloading datasets')
        datasets_info = {}

        if './downloaded_datasets/' not in os.listdir('.'):
            os.mkdir('./downloaded_datasets/')

        for dataset in tqdm(data['Dataset']):
            
            print(dataset)
            if (dataset + '.pkl') not in os.listdir('./downloaded_datasets/'):

                try:
                    ds_object = datasets.load_classification(
                        name=dataset,
                        return_metadata=True
                    )

                    with open(f'./downloaded_datasets/{dataset}.pkl', 'wb') as f:  # open a text file
                        pickle.dump(ds_object, f)

                    datasets_info[dataset] = ds_object
                except Exception as e:
                    print(f'The dataset {dataset} was not downloaded, due to {e}')
            else:
                with open(f'./downloaded_datasets/{dataset}.pkl', 'rb') as f:
                    datasets_info[dataset] = pickle.load(f) 



        print('Download completed')

        if 'downloaded_datasets' in os.listdir('.'):
            pass
        else:
            os.mkdir('./downloaded_datasets/')

        with open('./downloaded_datasets/datasets_info.pkl', 'wb') as f:  # open a text file
            pickle.dump(datasets_info, f) # serialize the list

        return datasets_info
    

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

            # model = used_model(
            #     X_train=X_train,
            #     X_test = X_test,
            #     y_train = y_train,
            #     y_test = y_test,
            #     metadata = metadata,
            #     random_state = random_state
            # )

            # if len(os.listdir('./model_checkpoints/' + model.model_folder)) != 0 :
            #     pass
            # else:
            #     model.training_process()

