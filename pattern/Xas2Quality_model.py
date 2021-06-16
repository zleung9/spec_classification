import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader

import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from torch.utils.tensorboard import SummaryWriter

class Xas2QualityFCN(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.4):
        super(Xas2QualityFCN, self).__init__()
        
        self.main = nn.Sequential(
            nn.BatchNorm1d(input_size, affine=False),
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(128, affine=False),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(32, affine=False),
            # nn.Linear(32,32),
            # nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            # nn.BatchNorm1d(32, affine=False),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(16, affine=False),
            nn.Linear(16,output_size),
            nn.Sigmoid()
        )   
    def forward(self, spec_in):
        return self.main(spec_in)

    @staticmethod
    def reset_weights(cls, m, verbose=False):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                if verbose: print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()


class Xas2QualityCNN(nn.Module):
    def __init__(self, output_size, dropout_rate=0.4):
        super(Xas2QualityCNN, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=2, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=2, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=2, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0),
            nn.Flatten(), # input shape for this layer is (*, 8, 4)
            
            nn.Dropout(p=dropout_rate),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(16,4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4,output_size),
            nn.Sigmoid()
        )

    def forward(self, spec_in):
        return self.main(spec_in)
    
