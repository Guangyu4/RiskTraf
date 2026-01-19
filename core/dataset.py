import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PEMSBDataset(Dataset):
    def __init__(self, data, indices, in_steps=12, out_steps=12):
        self.data = data
        self.indices = indices
        self.in_steps = in_steps
        self.out_steps = out_steps
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.data[t:t+self.in_steps]
        y = self.data[t+self.in_steps:t+self.in_steps+self.out_steps]
        return x, y


def load_data(file_path, in_steps=12, out_steps=12, train_ratio=0.6, val_ratio=0.2):
    data_dict = np.load(file_path)
    data = data_dict['data']
    
    num_nodes, num_timesteps, num_features = data.shape
    
    data = data.transpose(1, 0, 2)
    
    nan_count = np.isnan(data).sum()
    print(f"Initial NaN count: {nan_count} ({nan_count / data.size * 100:.2f}%)")
    
    data = np.where(np.isnan(data), 0, data)
    
    inf_count = np.isinf(data).sum()
    print(f"Inf count: {inf_count}")
    data = np.where(np.isinf(data), 0, data)
    
    train_data = data[:int(num_timesteps * train_ratio)]
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1, std)
    
    print(f"Mean shape: {mean.shape}, Mean values: {mean}")
    print(f"Std shape: {std.shape}, Std values: {std}")
    
    data = (data - mean) / std
    
    nan_after = np.isnan(data).sum()
    print(f"NaN count after normalization: {nan_after}")
    if nan_after > 0:
        data = np.where(np.isnan(data), 0, data)
    
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)
    
    total_steps = num_timesteps - in_steps - out_steps + 1
    train_steps = int(total_steps * train_ratio)
    val_steps = int(total_steps * val_ratio)
    
    train_indices = np.arange(train_steps)
    val_indices = np.arange(train_steps, train_steps + val_steps)
    test_indices = np.arange(train_steps + val_steps, total_steps)
    
    train_dataset = PEMSBDataset(data, train_indices, in_steps, out_steps)
    val_dataset = PEMSBDataset(data, val_indices, in_steps, out_steps)
    test_dataset = PEMSBDataset(data, test_indices, in_steps, out_steps)
    
    return train_dataset, val_dataset, test_dataset, mean, std, num_nodes


def get_dataloaders(file_path, batch_size=64, in_steps=12, out_steps=12, num_workers=4):
    train_dataset, val_dataset, test_dataset, mean, std, num_nodes = load_data(
        file_path, in_steps, out_steps
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, mean, std, num_nodes

