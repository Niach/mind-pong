import os
from dataclasses import dataclass

import numpy as np
from brainflow import DataFilter, FilterTypes
from torch import nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import torch.nn.functional as F

classes = ['REST', 'LEFT', 'RIGHT']


@dataclass
class EEGSample:
    data: np.ndarray
    state: int


def map_samples(sample):
    window_size = 250
    step_size = 200  # You can adjust this value to have overlapping or non-overlapping windows
    num_windows = (sample.data.shape[1] - window_size) // step_size + 1
    windows = []
    window_labels = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = sample.data[:, start_idx:end_idx].astype(np.float32)
        windows.append(window)
        window_labels.append(sample.state)

    return windows, window_labels


state_to_idx = {
    3: 0,  # REST
    1: 1,  # LEFT
    2: 2  # RIGHT
}


def plot_eeg(data, title, sample_rate=250.0):
    print(data.shape)
    print("dataT:")
    print(data.T)
    # plot eeg data over time
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(np.arange(data.shape[1]) / sample_rate, data.T)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [uV]')
    ax.set_title(title)
    plt.show()


# dataset class
class EEGDataset(Dataset):
    def __init__(self):
        self.classes = classes
        self.samples = []
        self.windows = []
        self.window_labels = []

        for csv_file in os.listdir('data/raw'):
            data = DataFilter.read_file(os.path.join('data/raw', csv_file))
            eeg_data = data[:-1, :]

            # plot_eeg(eeg_data, 'raw')
            eeg_data = self.apply_filters(eeg_data)
            # (eeg_data, 'filtered')

            marker_data = data[-1, :]
            # markers lye in the stream like this: 00..00100..00300..00200..00100..00300..00 where 1 indicated the start of a left movement,
            # 2 the start of a right movement and 3 the end of a movement (RESTing state)
            end_idx = None
            start_idx = 0
            for i in range(len(marker_data)):
                if marker_data[i] == 1 or marker_data[i] == 2:
                    if end_idx is not None:  # resting state
                        self.create_and_append_sample(eeg_data[:, start_idx:end_idx], 3)

                    start_idx = i
                elif marker_data[i] == 3:
                    end_idx = i
                    state = marker_data[start_idx]
                    if state != 0:
                        self.create_and_append_sample(eeg_data[:, start_idx:end_idx], int(state))

    def create_and_append_sample(self, eeg_data, state):
        sample = EEGSample(eeg_data, state_to_idx[int(state)])
        if len(sample.data[0]) > 250:
            self.samples.append(sample)
            wins, labs = map_samples(sample)
            self.windows.extend(wins)
            self.window_labels.extend(labs)

    def apply_filters(self, eeg_data):
        min_values = eeg_data.min(axis=1)[:, np.newaxis]
        max_values = eeg_data.max(axis=1)[:, np.newaxis]
        eeg_data = ((eeg_data - min_values) / (max_values - min_values)) * 200 - 100

        # Apply a notch filter at 50 Hz to remove powerline interference
        for i in range(eeg_data.shape[0]):
            DataFilter.perform_bandstop(eeg_data[i], 250, 48.0, 52.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

        # Apply a bandpass filter from 2-30 Hz to isolate the motor imagery related frequency components
        for i in range(eeg_data.shape[0]):
            DataFilter.perform_bandpass(eeg_data[i], 250, 2.0, 30.0, 4, FilterTypes.BUTTERWORTH.value, 0)

        return eeg_data

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data = self.windows[idx]
        label = self.window_labels[idx]
        return data, label


class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()

        # First block of conv layers
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(8, 1))
        self.batch_norm1 = nn.BatchNorm2d(25)
        self.pooling1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(0.5)

        # Second block of conv layers
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5))
        self.batch_norm2 = nn.BatchNorm2d(50)
        self.pooling2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(0.5)

        # Third block of conv layers
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5))
        self.batch_norm3 = nn.BatchNorm2d(100)
        self.pooling3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(0.5)

        # Fourth block of conv layers
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5))
        self.batch_norm4 = nn.BatchNorm2d(200)
        self.pooling4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout4 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(2200, 3)  # Update this line to match the shape of x
        # Adjust the size if your input shape is different

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding an extra dimension for channel

        # First block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Second block
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Third block
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        # Fourth block
        x = self.conv5(x)
        x = self.batch_norm4(x)
        x = F.elu(x)
        x = self.pooling4(x)
        x = self.dropout4(x)

        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
