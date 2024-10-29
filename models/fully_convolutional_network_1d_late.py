import torch
from torch import nn
from models.nn_model import NNModel

class FullyConvolutionalNetwork1DLate(NNModel):

    def __init__(self, train_dataset, test_dataset, metadata, model_name, random_state=42, device='cuda') -> None:
        super().__init__(train_dataset, test_dataset, metadata, model_name, random_state, device)

        self.network1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape_1[1], out_channels=128, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten()
        )