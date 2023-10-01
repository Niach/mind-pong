import os
from dataclasses import dataclass

import numpy as np
from brainflow import DataFilter, FilterTypes
from torch import nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
classes = ['REST', 'LEFT', 'RIGHT']


@dataclass
class EEGSample:
    data: np.ndarray
    state: int


def map_sample(sample, window_size=100):
    mid_point = len(sample.data[0]) // 2  # Assuming the data shape is (8, N)
    start_idx = max(0, mid_point - window_size // 2)
    end_idx = min(len(sample.data[0]), mid_point + window_size // 2)
    return (sample.data[:, start_idx:end_idx]).astype(np.float32)

state_to_idx = {
    3: 0,  # REST
    1: 1,  # LEFT
    2: 2   # RIGHT
}


def plot_eeg(data, title, sample_rate=250.0):
    time = np.arange(data.shape[1]) / sample_rate
    for channel in data:
        plt.plot(time, channel)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend([f'Channel {i+1}' for i in range(data.shape[0])])
    plt.show()


# dataset class
class EEGDataset(Dataset):
    def __init__(self):
        self.classes = classes
        self.samples = []

        for csv_file in os.listdir('data/raw'):
            with open(os.path.join('data/raw', csv_file), 'rb') as f:
                data = np.loadtxt(f, delimiter=',')
                eeg_data = data[:-1, :]

                plot_eeg(eeg_data, 'raw')
                eeg_data = self.apply_filters(eeg_data)
                plot_eeg(eeg_data, 'filtered')


                marker_data = data[-1, :]
                # markers lye in the stream like this: 00..00100..00300..00200..00100..00300..00 where 1 indicated the start of a left movement,
                # 2 the start of a right movement and 3 the end of a movement (RESTing state)
                end_idx = None
                for i in range(len(marker_data)):
                    if marker_data[i] == 1 or marker_data[i] == 2:
                        if end_idx is not None:  # resting state
                            self.create_and_append_sample(eeg_data[:, start_idx:end_idx], 3)

                        start_idx = i
                    elif marker_data[i] == 3:
                        end_idx = i
                        state = marker_data[start_idx]
                        self.create_and_append_sample(eeg_data[:, start_idx:end_idx], int(state))

    def create_and_append_sample(self, eeg_data, state):
        sample = EEGSample(eeg_data, state_to_idx[int(state)])
        if len(sample.data[0]) > 100:
            self.samples.append(sample)

    def apply_filters(self, eeg_data):
        # Apply a notch filter at 50 Hz to remove powerline interference
        for i in range(eeg_data.shape[0]):
            DataFilter.perform_bandstop(eeg_data[i], 250, 1.0, 30.0, 3,
                                        FilterTypes.BUTTERWORTH.value, 0)

        # Apply a bandpass filter from 8-30 Hz to isolate the motor imagery related frequency components
        for i in range(eeg_data.shape[0]):
            DataFilter.perform_bandstop(eeg_data[i], 250, 8.0, 30.0, 3, FilterTypes.BUTTERWORTH.value, 0)

        return eeg_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = map_sample(sample)
        label = sample.state
        return data, label


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))  # Convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))  # Convolutional layer
        self.fc1 = nn.Linear(32 * 8 * 100, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 3)  # Fully connected layer for 3 classes
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding an extra dimension for channel
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
