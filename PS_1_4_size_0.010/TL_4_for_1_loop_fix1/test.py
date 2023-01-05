# help function
from transfer_learning import NeuralNet
#from dataset_loader import data_loader, all_filter, get_descriptors, one_filter, data_scaler

# modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import os, sys
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing

from tqdm import tqdm
from scipy.stats import pearsonr

import matplotlib.pyplot as plt 
#%matplotlib inline

# file name and data path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = os.getcwd()
#file_name = 'CrystGrowthDesign_SI.csv'
print (device)
print ("Bingo!")


class SequenceDataset(Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)


X = np.load("../chain_double_clean.npy")
y = np.load("../F_double_clean.npy")
#X = np.load("../../../SEQ_PPI_24_16/chain_L24_maxmin.npy")
#y = np.load("../../../SEQ_PPI_24_16/PP_F_L24_maxmin.npy")
X = X[0:20000]
y = y[0:20000]


print(np.mean(y))
