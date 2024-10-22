import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# Step 1: Custom Dataset class (for multivariate time series)
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, metadata, device = 'cuda'):
        self.data = data
        self.labels = labels
        self.metadata = metadata
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a tuple of (input_data, label)
        return torch.tensor(self.data[idx], dtype=torch.float32).to(device=self.device), torch.tensor(self.labels[idx], dtype=torch.long).to(device=self.device)