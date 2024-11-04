import torch
from torch import nn
from models.nn_model import NNModel

class FullyConvolutionalNetwork1DIntermediate(NNModel):
    
    # def __init__(self, train_dataloader, test_dataloader, metadata, model_name = 'FullyConvolutionalNetwork1DIntermediate', random_state=42) -> None:
    #     super().__init__(train_dataloader, test_dataloader, metadata, 'FullyConvolutionalNetwork1DIntermediate', random_state, True)

    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'FullyConvolutionalNetwork1DIntermediate', random_state=42, device='cuda', is_multimodal=True) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal)

        self.modalities = nn.ModuleDict()
        self.num_modalities = len(self.input_shape)
        
        # Define sub-networks for each modality
        for i, input_shape in enumerate(self.input_shape):
            self.modalities[f"modality_{i}"] = nn.Sequential(
                nn.Conv1d(in_channels=input_shape[1], out_channels=128, kernel_size=8),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
                nn.ReLU(),
                nn.BatchNorm1d(256)
            )
        
        self.last_conv1d = nn.Conv1d(in_channels= self.num_modalities*256, out_channels=128, kernel_size=3)
        

        self.seq_after_conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten()
        )

        self.final_layer = nn.Linear(in_features=128, out_features=self.classes_shape[1])

    

    def forward(self, x):

        x_modalities = self.transform_multimodal(x)

        modality_outputs = []

        for i, input_shape in enumerate(self.input_shape): 
            modality_outputs.append(self.modalities[f"modality_{i}"](x_modalities[i]))

        
        out = torch.cat(modality_outputs, dim = 1)
        out = self.last_conv1d(out)
        out = self.seq_after_conv(out)
        out = self.final_layer(out)


        return out