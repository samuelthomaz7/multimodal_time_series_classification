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

    def __init__(self, in_channels, num_classes, depth, use_gap, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.depth = depth
        self.use_gap = use_gap


        self.inception_block = InceptionBlock(
            in_channels = in_channels, 
            out_channels= 32, 
            depth= self.depth,
            residual=False)
        
        self.num_classes = num_classes


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_block(x)
        
        if self.use_gap:
            x = torch.mean(x, dim=-1)
        
        return x
    

class InceptionTimeIntermediate(NNModel):

    # def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'InceptionTimeEnsembleIntermediate', random_state=42, device='cuda', is_multimodal=True) -> None:
    #     super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal)
        
    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'InceptionTimeIntermediate', random_state=42, device='cuda', is_multimodal=True, is_ensemble = True, model_num = None) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal, is_ensemble, model_num)
        
        self.num_models_ensemble = 5

        self.modalities = nn.ModuleDict()
        self.num_modalities = len(self.input_shape)
        # self.last_linear_layer = nn.ModuleList()
        # self.final_inception_modules = nn.ModuleList()

        # self.linear = nn.Linear(32 * 4, 1 if num_classes == 2 else num_classes)
        # for model_num in range(self.num_models_ensemble):
        
            

        for i, input_shape in enumerate(self.input_shape):
            
            # if f"modality_{i}" not in self.modalities.keys():
            #     self.modalities[f"modality_{i}"] = nn.ModuleList()
            # else:
            #     pass

            self.modalities[f"modality_{i}"] = InceptionTime(
                in_channels = input_shape[1],
                num_classes = self.classes_shape[1],
                depth = 4,
                use_gap=False

            )

        self.final_inception_modules = InceptionTime(
            in_channels = 128*self.num_modalities,
            num_classes = self.classes_shape[1],
            depth = 4,
            use_gap=True

        )
        
        self.last_linear_layer = nn.Linear(128, 1 if self.classes_shape[1] == 2 else self.classes_shape[1])
            


            


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_modalities = self.transform_multimodal(x)

        # output_list = []
        # for model_num in range(self.num_models_ensemble):
        
        modality_outputs = []
        
        for i, input_shape in enumerate(self.input_shape): 
            # x = self.modalities[f"modality_{i}"][model_num](x_modalities[i])
            x = self.modalities[f"modality_{i}"](x_modalities[i])
            modality_outputs.append(x)

        
        out = torch.cat(modality_outputs, dim = 1)
        # out = self.final_inception_modules[model_num](out)
        out = self.final_inception_modules(out)
        
        # out = self.last_linear_layer[model_num](out)
        out = self.last_linear_layer(out)

        # output_list.append(out)
                

        return out




