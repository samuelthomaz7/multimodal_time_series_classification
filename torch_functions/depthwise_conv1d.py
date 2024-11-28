import torch 
from torch import nn

class DepthwiseSeparableConvolution1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 2) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, padding='same', bias=False,
                                   dilation=dilation, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, padding='same', bias=False,
                                   dilation=dilation, stride=stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))