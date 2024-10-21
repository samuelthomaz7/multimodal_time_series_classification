import torch
import os
import pickle
from torch import nn
import time
from utils_file import set_seeds


class NNModel(nn.Module):

    def __init__(self, X_train, X_test, y_train, y_test, metadata, model_name, random_state = 42) -> None:
        super().__init__()

        set_seeds(random_state)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata = metadata
        self.random_state = random_state
        self.model_name = model_name
        self.epochs = 5000
        self.num_classes = self.metadata['class_values']
        self.batch_size = 32
        self.metrics = {}

        self.model_folder = self.model_name + '_' + self.metadata['problemname'].replace(' ', '_') + '_' + str(self.random_state)

        if 'model_checkpoints' not in os.listdir('.'):
            os.mkdir('./model_checkpoints')
        
        if self.model_folder not in os.listdir('./model_checkpoints'):
            os.mkdir('./model_checkpoints/' + self.model_folder)


        if len(self.metadata['class_values']) > 2:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            lr = 0.001,
            params=self.parameters()
        )

    
    def forward(self, x):
        pass

    def fit(self):

        start_time = time.time()
        for epoch in range(self.epochs):
            self.train()

            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0


        end_time = time.time()
        self.metrics = {
            'traning_time': end_time - start_time
        }

