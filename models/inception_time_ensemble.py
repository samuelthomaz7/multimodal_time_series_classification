import torch 
from torch import nn

from models.nn_model import NNModel
from typing import Any, Tuple, List, Dict

def noop(x: torch.Tensor) -> torch.Tensor:
    return x


class InceptionModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = [40, 20, 10],
                 bottleneck: bool = True) -> None:
        super().__init__()
        
        self.kernel_sizes = kernel_size
        bottleneck = bottleneck if in_channels > 1 else False
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False, padding='same') if bottleneck else noop
        
        self.convolutions = nn.ModuleList([
            nn.Conv1d(out_channels if bottleneck else in_channels,
                      out_channels,
                      kernel_size=k,
                      padding='same',
                      bias=False) for k in self.kernel_sizes
        ])
        self.maxconv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1),
                                       nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', bias=False)])
        self.batchnorm = nn.BatchNorm1d(out_channels * 4)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x
        x = self.bottleneck(x)
        x = torch.cat([conv(x) for conv in self.convolutions] + [self.maxconv(x_)], dim=1)
        return self.activation(x)


class InceptionBlock(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 residual: bool = False,
                 depth: int = 3) -> None:
        super().__init__()
        
        self.residual = residual
        self.depth = depth
        
        self.activation = nn.ReLU()
        
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        
        for d in range(depth):
            self.inception.append(InceptionModule(
                in_channels=in_channels if d == 0 else out_channels * 4,
                out_channels=out_channels
            ))
            if self.residual and d % 3 == 2:
                c_in, c_out = in_channels if d == 2 else out_channels * 4, out_channels * 4
                self.shortcut.append(
                    nn.BatchNorm1d(c_in) if c_in == c_out else nn.Sequential(*[
                        nn.Conv1d(c_in, c_out, kernel_size=1, padding='same'),
                        nn.BatchNorm1d(c_out)
                    ])
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                res = x = self.activation(x + self.shortcut[d // 3](res))

        return x



class InceptionTime(nn.Module):

    def __init__(self, in_channels, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


        self.inception_block = InceptionBlock(
            in_channels = in_channels, 
            out_channels= 32, 
            depth=8,
            residual=False)
        self.linear = nn.Linear(32 * 4, 1 if num_classes == 2 else num_classes)
        # self.num_classes = num_classes


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_block(x)
        x = torch.mean(x, dim=-1)
        return self.linear(x)
    

class InceptionTimeEnsemble(NNModel):

    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'InceptionTimeEnsemble', random_state=42, device='cuda', is_multimodal=False) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal)
        
        self.num_models_ensemble = 5
        self.models = nn.ModuleList()

        for i in range(self.num_models_ensemble):

            self.models.append(InceptionTime(
                in_channels = self.input_shape[1],
                num_classes = self.classes_shape[1]

            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output_list = []
        for i in range(self.num_models_ensemble):
            output_list.append(self.models[i](x))
        
        out = torch.stack(output_list)

        return torch.mean(out, dim = 0)




