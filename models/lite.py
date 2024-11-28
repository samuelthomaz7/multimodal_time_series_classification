import numpy as np
import torch 
from torch import nn

from models.nn_model import NNModel
from typing import Any, Tuple, List, Dict
from torch_functions.depthwise_conv1d import DepthwiseSeparableConvolution1d



class LITE(NNModel):

    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'LITE', random_state=42, device='cuda', is_multimodal=False, is_ensemble=True, model_num=None) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal, is_ensemble, model_num)


        custom_kernels_sizes = [2, 4, 8, 16, 32, 64]
        custom_convolutions = []
        
        for ks in custom_kernels_sizes:
            filter_ = np.ones(shape=(1, self.input_shape[1], ks))
            indices_ = np.arange(ks)

            filter_[:, :, indices_ % 2 == 0] *= -1 # increasing detection filter

            custom_conv = nn.Conv1d(in_channels=self.input_shape[1], out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())
            
            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        for ks in custom_kernels_sizes:
            filter_ = np.ones(shape=(1, self.input_shape[1], ks))
            indices_ = np.arange(ks)
            
            filter_[:,:, indices_ % 2 > 0] *= -1 # decreasing detection filter
            
            custom_conv = nn.Conv1d(in_channels=self.input_shape[1], out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())
            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        for ks in custom_kernels_sizes[1:]:
            filter_ = np.zeros(shape=(1, self.input_shape[1], ks + ks // 2))
            x_mesh = np.linspace(start=0, stop=1, num=ks//4 + 1)[1:].reshape((-1, 1, 1))
            
            filter_left = x_mesh ** 2
            filter_right = filter_left[::-1]
            
            filter_left = np.transpose(filter_left, (1, 2, 0))
            filter_right = np.transpose(filter_right, (1, 2, 0))
            
            filter_[:, :, 0:ks//4] = -filter_left
            filter_[:, :, ks//4:ks//2] = -filter_right
            filter_[:, :, ks//2:3*ks//4] = 2 * filter_left
            filter_[:, :, 3*ks//4:ks] = 2 * filter_right
            filter_[:, :, ks:5*ks//4] = -filter_left
            filter_[:, :, 5*ks//4:] = -filter_right
            
            custom_conv = nn.Conv1d(in_channels=self.input_shape[1], out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())
            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        self.custom_convolutions = nn.ModuleList(custom_convolutions)
        self.custom_activation = nn.ReLU()

        n_convs = 3
        # Hidden Channels
        n_filters = 32
        # Kernel size
        inception_kernel_sizes = [40 // (2 ** i) for i in range(n_convs)]
        inception_convolutions = []


        for i in range(len(inception_kernel_sizes)):
            inception_convolutions.append(
                nn.Conv1d(
                    in_channels=self.input_shape[1], out_channels=n_filters, kernel_size=inception_kernel_sizes[i],
                    stride=1, padding='same', dilation=1, bias=False
                )
            )

        self.inception_convolutions = nn.ModuleList(inception_convolutions)
        self.inception_batchnorm = nn.BatchNorm1d(num_features=96)
        self.inception_activation = nn.ReLU()


        separable_convolutions = []
        separable_kernel_size = 40 // 2

        for i in range(2):
            dilation_rate = 2 ** (i + 1)
            separable_conv = DepthwiseSeparableConvolution1d(
                in_channels=113 if i == 0 else n_filters, out_channels=n_filters,
                kernel_size=separable_kernel_size // (2 ** i), dilation=dilation_rate
            )
            separable_convolutions.append(separable_conv)

        self.separable_convolutions = nn.ModuleList(separable_convolutions)
        self.separable_batchnorm = nn.BatchNorm1d(num_features=n_filters)
        self.separable_activation = nn.ReLU()

        self.linear = nn.Linear(in_features=32, out_features= 1 if self.classes_shape[1] == 2 else self.classes_shape[1])

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        feature_maps = []

        for inception_conv in self.inception_convolutions:
            x_ = inception_conv(x)
            feature_maps.append(x_)


        custom_feature_maps = []
        for custom_conv in self.custom_convolutions:
            x_ = custom_conv(x)
            custom_feature_maps.append(x_)

        custom_feature_maps = torch.concat(custom_feature_maps, dim=1) # Concatenate channel-wise
        custom_feature_maps = self.custom_activation(custom_feature_maps)

        feature_maps = torch.concat(feature_maps, dim=1) # Concatenate channel-wise again
        feature_maps = self.inception_batchnorm(feature_maps)
        feature_maps = self.inception_activation(feature_maps)


        feature_maps = torch.concat([custom_feature_maps, feature_maps], dim=1)

        for separable_conv in self.separable_convolutions:
            feature_maps = separable_conv(feature_maps) 
            feature_maps = self.separable_batchnorm(feature_maps)
            feature_maps = self.separable_activation(feature_maps)

        feature_maps = torch.mean(feature_maps, dim=-1)

        return self.linear(feature_maps)

