import torch 
from torch import nn

from models.nn_model import NNModel
from typing import Any, Tuple, List, Dict
from torch_functions.depthwise_conv1d import DepthwiseSeparableConvolution1d



class LITE(NNModel):

    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'LITE', random_state=42, device='cuda', is_multimodal=False, is_ensemble=False, model_num=None) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal, is_ensemble, model_num)