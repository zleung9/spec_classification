import torch
from torch import nn, tensor, optim
from torch.utils.data import Dataset, DataLoader, dataset

import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from torch.utils.tensorboard import SummaryWriter
class Xas2QualityDataset(Dataset):
    def __init__(self, spec, label, feature, energy_cut, verbose=False, balance=False):
        super(Xas2QualityDataset, self).__init__()
        
        self.verbose = verbose
        self.spec = spec[:, (feature>energy_cut[0])&(feature<energy_cut[1])]
        self.spec_label = label
        self.feature = feature[(feature>energy_cut[0])&(feature<energy_cut[1])]
        self.feature_size = self.feature.shape[0]
        self.output_size = self.spec_label.shape[0]
        self.balanced = balance

        # only select label<0.000001 or label>0.000009 as valid data
        select_label = (self.spec_label>1-1e-6) | (self.spec_label<0+1e-6)
        self.data = self.spec[select_label]
        self.label = self.spec_label[select_label]
        print("Orignal | Label_1: {:d}, Label_0: {:d}".format((self.label>1-1e-6).sum(),(self.label<1e-6).sum()))

        # balance and shuffle data so there are equaly amount of 0 and 1.
        if balance: 
            self.balance()
            self.balanced = True
        print("Feature size: {}".format(self.feature_size))
        assert self.data.shape[0] == self.label.shape[0]


    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx], self.label[idx]
        return sample

    
    def balance(self):
        "Sample equal amount of good and bad data from original spec, and shuffle them"
        label_1 = self.spec_label > 1-1e-6
        label_0 = self.spec_label < 0+1e-6
        N_label_1 = label_1.sum()
        N_label_0 = label_0.sum()
        
        if N_label_1 > N_label_0: # if more good data than bad data, sample euqal amount of good data
            one_zero_match = np.random.randint(N_label_1,size=N_label_0) 
            data = np.vstack((self.spec[label_1][one_zero_match], self.spec[label_0]))
            
        else: # if less good data than bad data, sample euqal amount of bad data
            one_zero_match = np.random.randint(N_label_0,size=N_label_1)
            data = np.vstack((self.spec[label_1],self.spec[label_0][one_zero_match]))

        # label 1 for first half of data, label 0 for second half
        label = np.zeros((data.shape[0], 1))
        label[:int(data.shape[0]/2)] = 1 # label the first half as 1
        
        # randomly shuffle data and labels
        shuffle = np.random.permutation(data.shape[0])
        self.data = data[shuffle]
        self.label = label[shuffle]
        
        self.feature_size = self.data.shape[1]
        self.output_size = self.label.shape[1]
            
        if self.verbose:
            print("Balanced | Label_1: {}; Label_0: {}".format(self.label.sum(),(1-self.label).sum()))

    def create_train_val_test(self):
        pass
    
    
    @classmethod
    def from_file(cls, spec_file, quality_file, energy_cut=(7600,8000), splits=[0.7,0.2,0.1], 
                  verbose=True, balance=False):

        # load normalized spec data from file
        with open(spec_file,'rb') as f:
            spec_norm = pickle.load(f)
        red_norm = spec_norm['spec']
        feature_grid = spec_norm['feature_grid']

        # load from file quality labels (1 or 0) for normalized spec data
        with open(quality_file,'rb') as f:
            result = pickle.load(f)
        labels = result['label']
        sets = result['set']

        # get rid of labels that fall in between [0.1, 0.9]
        label_select = (labels>=0.9) | (labels<=0.1)

        return Xas2QualityDataset(spec=red_norm[label_select], label=labels[label_select], feature=feature_grid, 
                                  energy_cut=energy_cut, verbose=verbose, balance=balance)

   
# def get_Xas2Quality_dataloaders():

#     for name in ['train', 'test', 'set']:
    

#     train_loader = Dataloader(ds_train, batch_size=batch_size, shuffle=True)
#     val_loader = Dataloader(ds_val, batch_size=batch_size, shuffle=True)    
#     test_loader = Dataloader(ds_test, batch_size=batch_size, shuffle=True)

#     return train_loader, val_loader, test_loader
        