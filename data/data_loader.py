import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultiModalDataset(Dataset):
    def __init__(self, rgb_data, hsi_data, sonar_data, labels, transform=None):
        self.rgb_data = rgb_data
        self.hsi_data = hsi_data
        self.sonar_data = sonar_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb = self.rgb_data[idx]
        hsi = self.hsi_data[idx]
        sonar = self.sonar_data[idx]
        label = self.labels[idx]

        if self.transform:
            rgb = self.transform(rgb)
            hsi = self.transform(hsi)
            sonar = self.transform(sonar)

        return {'rgb': rgb, 'hsi': hsi, 'sonar': sonar, 'label': label}

def load_data(batch_size=32):
    # Example: Replace with actual data loading logic
    rgb_data = np.random.rand(100, 3, 64, 64)  # Example RGB images
    hsi_data = np.random.rand(100, 10, 64, 64)  # Example HSI data
    sonar_data = np.random.rand(100, 1, 64, 64)  # Example Sonar data
    labels = np.random.randint(0, 5, 100)  # Example labels

    dataset = MultiModalDataset(rgb_data, hsi_data, sonar_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
