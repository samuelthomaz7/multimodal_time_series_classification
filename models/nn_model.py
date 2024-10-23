import copy
import numpy as np
import torch
import os
import pickle
from torch import nn
from torch.utils.data import DataLoader
import time
from utils_file import get_lr, set_seeds


class NNModel(nn.Module):

    def __init__(self, train_dataset, test_dataset, metadata, model_name, random_state = 42, device = 'cuda') -> None:
        super().__init__()

        set_seeds(random_state)
        self.device = device

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.metadata = metadata
        self.random_state = random_state
        self.model_name = model_name
        self.epochs = 5000
        self.num_classes = self.metadata['class_values']
        self.batch_size = 32
        self.metrics = {}

        
        self.train_dataload = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataload = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.input_shape = self.train_dataload.dataset.data.shape
        self.classes_shape = self.train_dataload.dataset.labels.shape

        self.model_folder = self.model_name + '_' + self.metadata['problemname'].replace(' ', '_') + '_' + str(self.random_state)

        if 'model_checkpoints' not in os.listdir('.'):
            os.mkdir('./model_checkpoints')
        
        if self.model_folder not in os.listdir('./model_checkpoints'):
            os.mkdir('./model_checkpoints/' + self.model_folder)


        if len(self.metadata['class_values']) > 2:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        pass

    def fit(self):

        self.optimizer = torch.optim.Adam(
            lr = 0.001,
            params=self.parameters()
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.75, 
            # patience= 10, 
            patience= int(self.epochs*0.05), 
            verbose=True
        )

        self.history = {
            'train_loss' : [],
            'test_loss' : [],
            'train_accuracy' : [],
            'test_accuracy' : [],
            'epochs': list(range(self.epochs))
        }

        best_val_acc = 0
        patience_early_stopping = int(self.epochs*0.5)
        patience_lr = int(self.epochs*0.05)

        start_time = time.time()

        for epoch in range(self.epochs):
            self.train()

            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_dataload):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs.type(torch.float32), targets.type(torch.float32))

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Calculate running loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == torch.max(targets, 1).indices).sum().item()
                total_predictions += targets.size(0)

            # Epoch loss and accuracy
            epoch_loss = running_loss / len(self.train_dataload)
            epoch_accuracy = correct_predictions / total_predictions

            self.eval()
            with torch.inference_mode():
                valid_loss = 0.0
                valid_correct = 0
                valid_total = 0
                for inputs, targets in self.test_dataload:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self(inputs)
                    loss = self.loss_fn(outputs.type(torch.float32), targets.type(torch.float32))

                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    valid_correct += (predicted == torch.max(targets, 1).indices).sum().item()
                    valid_total += targets.size(0)

                valid_loss /= len(self.test_dataload)
                valid_accuracy = valid_correct / valid_total
            
            self.scheduler.step(valid_accuracy)

            self.history['train_loss'].append(epoch_loss)
            self.history['test_loss'].append(valid_loss)
            self.history['train_accuracy'].append(epoch_accuracy)
            self.history['test_accuracy'].append(valid_accuracy)

            current_lr = get_lr(self.optimizer)
            
            if valid_accuracy >= 0.99999999:
                break

            if valid_accuracy > best_val_acc:
                best_val_acc = valid_accuracy
                # best_model_weights = copy.deepcopy(self.state_dict()) 

                torch.save(self.state_dict(), './model_checkpoints/' + self.model_folder + '/best_model.pth')

                patience_early_stopping = int(self.epochs*0.5)  
                patience_lr = int(self.epochs*0.05) 
            else:
                patience_early_stopping = patience_early_stopping - 1
                patience_lr = patience_lr - 1

                if patience_lr == 0:
                    patience_lr = int(self.epochs*0.05)

                if patience_early_stopping == 0:
                    break

            print(f'{self.random_state} - Epoch [{epoch+1}/{self.epochs}]| Loss: {epoch_loss:.4f}| Accuracy: {epoch_accuracy:.4f}| '
              f'Val Loss: {valid_loss:.4f}| Val Accuracy: {valid_accuracy:.4f} | LR: {current_lr} | Patience left ES: {patience_early_stopping} | Patience left LR: {patience_lr}')




        end_time = time.time()
        self.metrics = {
            'traning_time': end_time - start_time,
            'history': self.history
        }

        with open('./model_checkpoints/' + self.model_folder + '/metrics.pkl', 'wb') as f:  # open a text file
            pickle.dump(self.metrics, f) # serialize the list 



