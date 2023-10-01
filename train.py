
import time

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from inference import predict
from models import EEGSample, classes, map_sample, EEGDataset, EEGNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# This will be the early stopping function
def early_stopping(val_losses, curr_epoch, patience=20, delta=0.001):
    if len(val_losses) < patience:
        return False

    # Check if the validation loss has increased more than delta for patience epochs
    if all([(v - min(val_losses[-patience:])) > delta for v in val_losses[-patience:]]):
        return True

    return False


def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state_dict = None

    for epoch in range(epochs):
        # Training loop
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

        if early_stopping(val_losses, epoch):
            model.load_state_dict(best_state_dict)
            print(f'Early stopping on epoch {epoch}')
            break

    return model

def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')
    return 100 * correct / total



def train_run():
    dataset = EEGDataset()

    # Split ratios
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = EEGNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    model = train(model, train_loader, val_loader, criterion, optimizer, 500)
    acc = evaluate(model, test_loader)

    # Predict

    print("Testing speed")
    start = time.time()
    for i in range(50):
        inputs, labels = next(iter(test_loader))
        predicted = predict(model, inputs.to(device))
    end = time.time()
    print("avg inference time:", (end - start) / 50)
    return model, acc

if __name__ == '__main__':
    min_acc = 100
    max_acc = 0
    sum_acc = 0
    max_model = None
    for i in range(5):
        model, curr = train_run()
        if curr < min_acc:
            min_acc = curr
        if curr > max_acc:
            max_acc = curr
            max_model = model
        sum_acc += curr
    print("min:", min_acc)
    print("max:", max_acc)
    print("avg:", sum_acc / 5)
    torch.save(max_model.state_dict(), "model.pt")
    print("saved best model")
