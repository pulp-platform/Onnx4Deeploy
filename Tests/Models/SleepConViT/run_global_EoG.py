import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchmetrics
import os
from utils.data_loading import * 
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import socket
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from utils.gpu_utils import *
    
class EpiDeNet(nn.Module):
    def __init__(self, output_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,4), stride = (1,1), padding = 'same')
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,8), stride=(1,8))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1,16), stride = (1,1), padding = 'same')
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,8), stride = (1,1), padding = 'same')
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        # Adjusted kernel size for pool4 to avoid reducing the height
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(16,1), stride=(1,1), padding='same')
        self.bn4 = torch.nn.BatchNorm2d(16)
        # Adjusted pooling to maintain the height
        self.pool4 = nn.MaxPool2d(kernel_size=(1,1), stride=(4,1))
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(8,1), stride=(1,1), padding='same')
        self.bn5 = torch.nn.BatchNorm2d(16)
        self.pool6 = nn.AdaptiveAvgPool2d((1,1))    
        self.flatten = nn.Flatten()
        self.fcn = nn.Linear(16, output_classes)

    
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x= self.pool1(x)
        x=F.relu(self.bn2(self.conv2(x))) 
        x= self.pool2(x)
        x=F.relu(self.bn3(self.conv3(x)))
        x= self.pool3(x)
        x=F.relu(self.bn4(self.conv4(x)))
        x= self.pool4(x)
        x=F.relu(self.bn5(self.conv5(x)))
        x= self.pool6(x)
        x = self.flatten(x)
        x= self.fcn(x)
        return x
    
def train(model, device, train_loader, optimizer, criterion):
    model.train()  # Set the model to training mode
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to the correct device
        optimizer.zero_grad()  # Zero out the gradients to avoid accumulation
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        total_loss += loss.item()  # Accumulate loss

    avg_loss = total_loss / len(train_loader)  # Optional: Compute average loss for monitoring
    return avg_loss

def test(model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    total_samples = 0
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=11).to(device)  # Create an accuracy metric instance

    with torch.no_grad():  # No need to track gradients
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to the correct device
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Compute loss
            test_loss += loss.item() * data.size(0)  # Adjust loss by batch size for accuracy
            total_samples += data.size(0)
            pred = output.argmax(dim=1)  # Get the index of the max log-probability
            accuracy_metric.update(pred, target)  # Update accuracy metric with current batch

    avg_test_loss = test_loss / total_samples  # Compute average test loss
    accuracy = accuracy_metric.compute() * 100  # Compute accuracy percentage

    # Return average test loss along with accuracy for more detailed monitoring
    return avg_test_loss, accuracy

def load_eog_data(data_path, method='moving_average', apply_notch_filter=True, apply_bandpass_filter=True, window_size=2, subjects_exclude=None, zero_pad = False):
    data_all = None
    label_all = None

    subjects = os.listdir(data_path)
    for subject in subjects:
        if subjects_exclude is not None and subject in subjects_exclude:
            continue
        sessions = os.listdir(os.path.join(data_path, subject))
        for session in sessions:
            session_path = os.path.join(data_path, subject, session)
            data_new, label_new = load_real_data_new_new(session_path, sampling_freq=500, window_size=window_size, 
                                                     method=method, zero_pad=zero_pad,notch_filter=apply_notch_filter, 
                                                     bandpass_filter=apply_bandpass_filter)
            if data_all is None:
                data_all = data_new
                label_all = label_new
            else:
                data_all = np.concatenate((data_all, data_new), axis=2)
                label_all = np.concatenate((label_all, label_new), axis=0)
    
    data_all = np.swapaxes(data_all, 1, 2)  # Swap axes
    data_all = np.swapaxes(data_all, 0, 1)
    data_all = np.expand_dims(data_all, axis=2)

    return data_all, label_all


hostname = socket.gethostname()
path = '/home/thoriri/Documents/brainmetrics/gpu_status/'
gpu_size = 4
if(hostname + '_gpu_status.txt' not in os.listdir(path)):
    with open(path + hostname + '_gpu_status.txt', 'w') as f:
        for i in range(gpu_size):
            f.write(f"{i},available\n")
data_path = '/scratch/thoriri/EOG_data/csv/'
method = 'matlab'
apply_notch_filter = False
apply_bandpass_filter = True
patients = os.listdir(data_path)
num_epochs = 200
num_trials = 10  # Number of trials per fold with different random states

# Initialize a DataFrame to store the best accuracies for each patient
data_entries = []
try:
    gpu_id = get_and_lock_gpu(path + hostname + '_gpu_lock.lock', path + hostname + '_gpu_status.txt')
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Using GPU {gpu_id}")
    else:
        raise RuntimeError("No available GPU found.")
    for apply_notch_filter in [True, False]:
        for apply_bandpass_filter in [True, False]:
            for method in ['moving_average', 'matlab']:
                counter = 0
                for patient in patients:
                    print(f"Processing patient {patient}...")
                    sub_excluded = [p for p in patients if p != patient]
                    data_all_included, label_all_included = load_eog_data(data_path, method, apply_notch_filter, apply_bandpass_filter, 2, sub_excluded)
                    if(counter == 0):
                        data_all = data_all_included
                        label_all = label_all_included
                    else:
                        data_all = np.concatenate((data_all, data_all_included), axis=0)
                        label_all = np.concatenate((label_all, label_all_included), axis=0)
                    counter += 1
                X_tensor = torch.tensor(data_all, dtype=torch.float32)
                y_tensor = torch.tensor(label_all, dtype=torch.long)
                X_tensor = X_tensor.permute(0, 2, 1, 3)
                dataset = TensorDataset(X_tensor, y_tensor)
                fold_accuracies = []

                for trial in range(num_trials):
                    random_state = np.random.randint(0, 1000)
                    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
                    trial_fold_accuracies = []

                    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
                        train_subsampler = Subset(dataset, train_idx)
                        test_subsampler = Subset(dataset, test_idx)
                        train_loader = DataLoader(train_subsampler, batch_size=256, shuffle=True)
                        test_loader = DataLoader(test_subsampler, batch_size=256, shuffle=False)
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = EpiDeNet(output_classes=11).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=0.001)
                        criterion = nn.CrossEntropyLoss()
                        epoch_accuracies = []

                        for epoch in range(num_epochs):
                            train(model, device, train_loader, optimizer, criterion)
                            _, accuracy = test(model, device, test_loader, criterion)
                            epoch_accuracies.append(accuracy)
                        epoch_accuracies = torch.tensor(epoch_accuracies).to('cpu').numpy()
                        trial_fold_accuracies.append(epoch_accuracies)

                    # Averaging accuracies over folds within the same trial
                    trial_mean_accuracies = np.mean(trial_fold_accuracies, axis=0)
                    fold_accuracies.append(trial_mean_accuracies)

                # Calculate average accuracy over all trials and folds for each epoch
                mean_accuracies = np.mean(fold_accuracies, axis=0)
                max_acc = max(mean_accuracies)
                best_epoch = np.argmax(mean_accuracies)
                # Collect data for each patient
                data_entries.append({'Patient': 'All', 'Best Accuracy': max_acc, 'Best Epoch': best_epoch, 'Method': method, 'Notch Filter': apply_notch_filter, 'Bandpass Filter': apply_bandpass_filter})

    # Create a DataFrame from the collected data entries
    results_df = pd.DataFrame(data_entries)
    # Save the results to CSV file
    results_df.to_csv('results/global.csv', index=False)
    if gpu_id is not None:
        print("Releasing GPU")
        release_gpu(gpu_id, path + hostname + '_gpu_lock.lock', path + hostname + '_gpu_status.txt')
except Exception as e:
    if gpu_id is not None:
        print("Releasing GPU due to exception")
        release_gpu(gpu_id, path+hostname + '_gpu_lock.lock', path+hostname + '_gpu_status.txt')
    raise e
    