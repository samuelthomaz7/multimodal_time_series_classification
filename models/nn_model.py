import torch
import os
import pickle
from torch import nn


class NNModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)