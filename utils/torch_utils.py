# torch_utils.py
# SUMMARY: utility functions to reduce clutter in the main model classes

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import sampler
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch
# END IMPORTS

# CONSTANTS
PERCENT_TRAIN = 0.80
# END CONSTANTS

# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential
class Flatten(nn.Module):
    def flatten(x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, x):
        return self.flatten(x)


def process_data(data, labels, exclude_labels=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = torch.from_numpy(data).mean()
    std = torch.from_numpy(data).std()
    transform_norm = T.Compose([
        T.Normalize(mean, std)
    ])
    X = transform_norm(torch.from_numpy(data))

    new_data = []
    for i in range(len(X)):
        if exclude_labels != None:
            if labels[i] not in exclude_labels: 

                new_data.append([X[i], labels[i]])

    new_labels = np.array(new_data)[:,1]
    print("num 0s [Pittsburgh]: ", np.sum(new_labels == 0))
    print("num 1s [Orlando]: ", np.sum(new_labels == 1))
    print("num 2s [New York]: ", np.sum(new_labels == 2))    
    # CONSTRUCT data loaders
    NUM_TRAIN = int(PERCENT_TRAIN * len(new_data))
    np.random.shuffle(new_data)
    NUM_TOTAL = len(new_data)

    loader_val = DataLoader(new_data, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TOTAL)))
    loader_train = DataLoader(new_data, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    return loader_train, loader_val

def process_regression(data, labels, exclude_labels=None):
    mean = torch.from_numpy(data).mean()
    std = torch.from_numpy(data).std()
    transform_norm = T.Compose([
        T.Normalize(mean, std)
    ])

    X = transform_norm(torch.from_numpy(data))
    new_data = []
    remove_count = 0
    for i in range(len(X)):
        if exclude_labels != None:
            if labels[i][0] < 34 and (2 not in exclude_labels): 
                new_data.append([X[i], labels[i]])
            elif(labels[i][1] > -75 and (0 not in exclude_labels)):
                new_data.append([X[i], labels[i]])
            elif(labels[i][1] < -75 and (1 not in exclude_labels)):
                new_data.append([X[i], labels[i]])
            else: remove_count += 1
    print("{NUM} instances removed: ".format(NUM=remove_count))
    print("remaining data: ", len(new_data))

    # CONSTRUCT data loaders
    NUM_TRAIN = int(PERCENT_TRAIN * len(new_data))
    NUM_TOTAL = len(new_data)

    loader_val = DataLoader(new_data, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TOTAL)))
    loader_train = DataLoader(new_data, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    return loader_train, loader_val


def process_partition(data, labels, exclude_labels=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = torch.from_numpy(data).mean()
    std = torch.from_numpy(data).std()
    transform_norm = T.Compose([
        T.Normalize(mean, std)
    ])
    X = transform_norm(torch.from_numpy(data))

    new_data = []
    for i in range(len(X)):
        if exclude_labels != None:
            if labels[i] not in exclude_labels: 
                new_data.append([X[i], labels[i]])

    new_labels = np.array(new_data)[:,1]
    print("num 0s [East]: ", np.sum(new_labels == 0))
    print("num 1s [West]: ", np.sum(new_labels == 1))
    # CONSTRUCT data loaders
    NUM_TRAIN = int(PERCENT_TRAIN * len(new_data))
    np.random.shuffle(new_data)
    NUM_TOTAL = len(new_data)

    loader_val = DataLoader(new_data, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TOTAL)))
    loader_train = DataLoader(new_data, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    return loader_train, loader_val
