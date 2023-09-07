import os
import pickle
from dataclasses import dataclass

import numpy as np
from torch import nn
from torch.utils.data import Dataset

classes = ['REST', 'LEFT', 'RIGHT']


@dataclass
class EEGData:
    channelData: list[float]
    timestamp: int


@dataclass
class EEGSample:
    data: list[EEGData]
    timestamp_start: int
    timestamp_end: int
    state: str


def map_sample(sample):
    # Calculating the start and end indices for the middle 100 EEGData objects
    mid_point = len(sample.data) // 2
    start_idx = mid_point - 50
    end_idx = mid_point + 50

    # Extracting the central 100 EEGData objects
    central_data_objects = sample.data[start_idx:end_idx]

    data_list = [data.channelData for data in central_data_objects]

    data = np.array(data_list).astype(np.float32)

    # Ensure this reshape is compatible with your model's input
    # Since each channelData has length 70, and you're taking 100 EEGData objects, shape will be (100, 70)
    data = np.reshape(data, (100, 70))
    return data


# dataset class
class EEGDataset(Dataset):
    def __init__(self):
        self.classes = classes
        self.samples = []

        for c in self.classes:
            p = os.path.join('data/raw', c)

            for pickleSample in os.listdir(p):
                with open(os.path.join(p, pickleSample), 'rb') as f:
                    sample: EEGSample = pickle.load(f)
                    self.samples.append(sample)
        print('loaded', len(self.samples), 'samples')




    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = map_sample(sample)
        label = classes.index(sample.state)
        return data, label



class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 9))
        self.conv2 = nn.Conv2d(16, 32, (70, 1))
        self.fc1 = nn.Linear(61504, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, 100, 70)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

