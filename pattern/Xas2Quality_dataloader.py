import pickle
import math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import WeightedRandomSampler


from torch.utils.tensorboard import SummaryWriter
class Xas2QualityDataset(Dataset):
    def __init__(self, spec, label, feature, energy_cut, verbose=False):
        super(Xas2QualityDataset, self).__init__()
        
        self.verbose = verbose

        # select a certain range of features for training
        feature_range = (feature>energy_cut[0])&(feature<energy_cut[1])
        self.feature = feature[feature_range]
        self.feature_size = self.feature.shape[0]

        # only select label<0.001 or label>0.009 as valid data
        self.spec = spec[:, feature_range]
        self.spec_label = label.reshape(-1,1)
        select_label = (self.spec_label>=0) & ((1-1e-3<self.spec_label)|(self.spec_label<0+1e-3))
        self.data = self.spec[select_label.flatten()] # flatten() reduces dimension from 2 to 1
        self.label = self.spec_label[select_label.flatten()]
        self.sample_size = self.label.shape[0]
        assert self.label.shape[1] == 1 # assert self.label has two dimensions

        if verbose:
            print("Orignal Label\n1: {:d}, 0: {:d}, -1: {:d}"
                  .format((self.label>0.5).sum(),(self.label<0.5).sum(),(self.spec_label==-1).sum()))
        
        # create a weight sequence of good and  d data according to their fractions
        good_mask, bad_mask = self.label>0.5, self.label<0.5
        self.weights = good_mask * bad_mask.sum() + bad_mask * good_mask.sum()
        
        # data is not partitioned into train, test, validation sets by default
        self.has_partition = False


    def __len__(self):
        return self.data.shape[0]
    

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx], self.label[idx]
        return sample

    
    def partition(self, ratio=(0.7, 0.15, 0.15)):
        '''
        Partition the dataset into train, validation and test set according to ratio.
        '''
        N_samples = self.sample_size

        N_train = int(np.floor(N_samples * ratio[0]))
        N_test = int(np.floor(N_samples * ratio[1]))
        N_val = N_samples - N_train - N_test

        index_shuffle = np.random.permutation(range(N_samples)) # shuffle data and label
        self.index_train = index_shuffle[:N_train]
        self.index_test = index_shuffle[N_train: N_train+N_test]
        self.index_val = index_shuffle[-N_val:]

        self.has_partition = True
    

    @classmethod
    def from_file(cls, spec_file, quality_file, energy_cut=(7600,8000), splits=[0.7,0.2,0.1], 
                  verbose=True):

        # load normalized spec data from file
        with open(spec_file,'rb') as f:
            spec_norm = pickle.load(f)
        red_norm = spec_norm['spec']
        feature_grid = spec_norm['feature_grid']

        # load from file quality labels (1 or 0) for normalized spec data
        with open(quality_file,'rb') as f:
            result = pickle.load(f)
        labels = result['label']
        sets = np.array(result['set'])
        labels[sets=='test'] = -1 # unlabeled data has label -1

        # # get rid of labels that fall in between [0.1, 0.9]
        # label_select = (labels>=0.9) | (labels<=0.1)

        return Xas2QualityDataset(spec=red_norm, label=labels, feature=feature_grid,
                                  energy_cut=energy_cut, verbose=verbose)

   

def get_Xas2Quality_dataloaders(dataset, batch_size, ratio=(0.7, 0.15, 0.15), load_unlabel=False):
    
    # partition the totald dataset into train,validation and test sets according to ratio
    dataset.partition(ratio=ratio)
    if not dataset.has_partition: # has_partion attribute is only turned on after partition
        print("Data partition failed.")
        return None, None, None
    else:
        ds_train = Subset(dataset, dataset.index_train)
        ds_test = Subset(dataset, dataset.index_test)
        ds_val = Subset(dataset, dataset.index_val)

    train_sampler = WeightedRandomSampler(dataset.weights[dataset.index_train], replacement=True,
                                          num_samples=math.ceil(len(ds_train)/batch_size)*batch_size)

    train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=0, pin_memory=False,
                              sampler=train_sampler)
    val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=0, pin_memory=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=0, pin_memory=False)

    # create a dataloader for unlabeled data for extreme prob. reduction (see Trainer.train())
    if load_unlabel: 
        index_all = np.arange(len(dataset.spec))
        select_unlabeled = (dataset.spec_label==-1).flatten()
        index_unlabeled = index_all[select_unlabeled]
        ds_unlabeled = Subset(dataset, index_unlabeled)
        unlabel_loader = DataLoader(ds_unlabeled, batch_size=batch_size, num_workers=0, pin_memory=False)
        return train_loader, val_loader, test_loader, unlabel_loader

    return train_loader, val_loader, test_loader 