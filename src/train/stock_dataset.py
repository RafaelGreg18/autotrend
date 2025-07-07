import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, path_csv, sequence_length=30):
        self.data = pd.read_csv(path_csv)
        self.sequence_length = sequence_length
        
        self.X = self.data.iloc[:, :-1].values.astype('float32')
        # keep y as int (0 or 1)
        self.y = self.data.iloc[:, -1].values.astype('int')
        
    def __len__(self):
        # Adjust length if using sequences
        if self.sequence_length > 1:
            return len(self.data) - self.sequence_length + 1
        return len(self.data)

    def __getitem__(self, idx):
        if self.sequence_length > 1:
            # Return a sequence of data
            x = torch.tensor(self.X[idx:idx+self.sequence_length], dtype=torch.float32)
            # Target is the last element in sequence
            y = torch.tensor(self.y[idx+self.sequence_length-1], dtype=torch.long)
        else:
            # Return a single data point
            x = torch.tensor(self.X[idx], dtype=torch.float32)
            y = torch.tensor(self.y[idx], dtype=torch.long)
        
        return x, y