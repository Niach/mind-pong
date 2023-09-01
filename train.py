import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

classes = ['REST', 'LEFT', 'RIGHT']


# one input line consists of 70 floats (-100 to 100) (generator function)
def generate_data():
    while True:
        yield [random.uniform(-100, 100) for _ in range(70)]

def generate_label():
    while True:
        yield random.randint(0, 2)


# consume data from generator. it represents a 25hz signal with 70 channels. make 2s windows without overlap
# Here, it is important to note that data must be in channels x samples DataFrame format. As EEG data is often presented with the electrode channels as columns, do not forget to transpose your data before.
def generate_sample():
    data = generate_data()
    while True:
        d = np.array([next(data) for _ in range(50)]).transpose()
        l = next(generate_label())
        yield d, l



# dataset class
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx])


def generate_dataset():
    data = []
    labels = []
    for _ in range(1000):
        d, l = next(generate_sample())
        data.append(d)
        labels.append(l)
    return EEGDataset(data, labels)


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 9))
        self.conv2 = nn.Conv2d(16, 32, (70, 1))
        self.fc1 = nn.Linear(32 * 42, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, 70, 50)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x



def train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

def predict(model, inputs):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted_indices = torch.max(outputs.data, 1)
    return [classes[idx] for idx in predicted_indices]


if __name__ == '__main__':
    dataset = generate_dataset()

    # Split ratios
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = EEGNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, 10)
    evaluate(model, test_loader)

    # Predict
    inputs, labels = next(iter(test_loader))
    predicted = predict(model, inputs)
    print(f'Predicted: {predicted}')

