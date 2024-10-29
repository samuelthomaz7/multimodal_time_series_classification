import torch
from torch import nn

from models.nn_model import NNModel

class MultiLayerPerceptron(NNModel):

    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'MultiLayerPerceptron', random_state=42, device='cuda', is_multimodal=False) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal)

        self.network = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(in_features = self.input_shape[1]*self.input_shape[2], out_features= 500),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            
            nn.Linear(in_features = 500, out_features= 500),
            nn.ReLU(),
            nn.Dropout(p = 0.2),

            nn.Linear(in_features = 500, out_features = 500),
            nn.Dropout(p = 0.3),
            nn.ReLU(),

            nn.Linear(in_features = 500, out_features= self.classes_shape[1]),
            # nn.Softmax()
        )


    def forward(self, x):
        return self.network(x)

        



        # input = nn.Linear(in_features=)