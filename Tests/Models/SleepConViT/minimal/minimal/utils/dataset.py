from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
    

class EOGDataset(Dataset):
    def __init__(self, data_path, labels_path, fewer_channels=False, seb_data = False, num_samples = 1000):
        """
        Initializes the dataset.
        """
        self.eegs = np.load(data_path)  # Assuming shape is (8, 1000, num_samples)
        self.labels = np.load(labels_path)
        self.fewer_channels = fewer_channels
        self.seb_data = seb_data
        self.num_samples = num_samples
        
        # Create a mapping for labels to convert them from [50, 51, ..., 80] to [0, 1, ..., 10]
        unique_labels = np.unique(self.labels)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.eegs.shape[2]
    
    def __getitem__(self, index):
        """
        Retrieves a single sample and its label.
        """
        X = self.eegs[:, :, index].transpose((1, 0))  # Shape is (1000, 8)
        X = X[:self.num_samples,:]
        # Select only the first three channels
        if(not self.fewer_channels):
            X = X[:, [0,1,2]]
        if(not self.seb_data):
            # Clip the data between -1024 and 1024, normalize by dividing by 32.0
            X = np.clip(X, -1024, 1024) / 32.0
            X = np.nan_to_num(X, nan=0)
        
        
        # Fetch the label and map it
        y = self.labels[index]
        y_mapped = self.label_mapping[y]
        
        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y_mapped = torch.tensor(y_mapped, dtype=torch.long)
        
        return X, y_mapped
    


class EOGDataset_ss(Dataset):
    def __init__(self, data_path, subjects, num_samples = 1000):
        """
        Initializes the dataset.
        """
        self.subjects = subjects # This is a list of subjects such as ['01', '02']
        self.num_samples = num_samples
        # Load the data for all the subjects
        self.eegs = []
        self.labels = []
        for subject in subjects:
            data_path_subject = data_path + f'sub{subject}_data.npy'
            labels_path_subject = data_path + f'sub{subject}_labels.npy'
            eeg = np.load(data_path_subject)
            label = np.load(labels_path_subject)
            self.eegs.append(eeg)
            self.labels.append(label)
        # Concatenate the data
        self.eegs = np.concatenate(self.eegs, axis=2)
        self.labels = np.concatenate(self.labels, axis=0)
        # Create a mapping for labels to convert them from [50, 51, ..., 80] to [0, 1, ..., 10]
        unique_labels = np.unique(self.labels)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.eegs.shape[2]
    
    def __getitem__(self, index):
        """
        Retrieves a single sample and its label.
        """
        X = self.eegs[:, :, index].transpose((1, 0))  # Shape is (1000, 8)
        X = X[:self.num_samples,:]
        # Select only the first three channels
        X = X[:, [0,1,2]]
        # Clip the data between -1024 and 1024, normalize by dividing by 32.0
        X = np.clip(X, -1024, 1024) / 32.0
        X = np.nan_to_num(X, nan=0)
        
        
        # Fetch the label and map it
        y = self.labels[index]
        y_mapped = self.label_mapping[y]
        
        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y_mapped = torch.tensor(y_mapped, dtype=torch.long)
        
        return X, y_mapped