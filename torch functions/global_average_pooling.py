import torch


class GlobalAveragePooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=-1)