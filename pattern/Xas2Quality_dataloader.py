import pickle
import math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


from torch.utils.tensorboard import SummaryWriter
class Xas2QualityDataset(Dataset):
    def __init__(self, spec, label, feature, energy_cut, verbose=False):
        '''
        DOC
        '''

        super(Xas2QualityDataset, self).__init__()
        assert label.shape[1] == 1 # label should have shape (N,1)
        self.verbose = verbose

        # select a certain range of features for training
        feature_range = (feature>energy_cut[0])&(feature<energy_cut[1])
        self.feature = feature[feature_range]
        self.feature_size = self.feature.shape[0]
        self.data = spec[:, feature_range]
        self.label = label
        self.sample_size = self.label.shape[0]
        self.weights = np.ones(self.sample_size) # weights are set to 1 by default

        if verbose:
            select_unlabeled = self.label.flatten()==-1 # label = -1
            select_good = 1-1e-3<self.label.flatten() # label = 1
            select_bad = (self.label.flatten()<0+1e-3) & (self.label.flatten()>=0) # label = 0
            print("Orignal Label\n1: {:d}, 0: {:d}, -1: {:d}"
                  .format(select_good.sum(),select_bad.sum(),select_unlabeled.sum()))
        
        self.has_partition = False # has_partition flag is set to False by default

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
        select_unlabeled = self.label.flatten()==-1 # label = -1
        select_good = 1-1e-3<self.label.flatten() # label = 1
        select_bad = (self.label.flatten()<0+1e-3) & (self.label.flatten()>=0) # label = 0
        select_labeled = select_good | select_bad # label = 0 or 1
        assert select_unlabeled.sum()+select_labeled.sum() == self.sample_size # exhaustive test

        index_total = np.arange(self.sample_size)
        index_labeled = index_total[select_labeled] # select out the index of labeled data
        index_unlabeled = index_total[select_unlabeled] # select out the index of unlabeled data

        N_labeled = select_labeled.sum() # number of labeled data
        N_train = int(np.floor(N_labeled * ratio[0])) # size of train set
        N_test = int(np.floor(N_labeled * ratio[1])) # size of test set
        N_val = N_labeled - N_train - N_test # size of val set

        index_shuffle = np.random.permutation(index_labeled) # shuffle labeled data and label
        self.index_train = index_shuffle[:N_train]
        self.index_test = index_shuffle[N_train: N_train+N_test]
        self.index_val = index_shuffle[-N_val:]
        self.index_unlabeled = index_unlabeled

        # create a weight list to be used by WeightedRandomSampler
        self.weights = select_good/select_good.sum() + select_bad/select_bad.sum() \
                     + select_unlabeled/select_unlabeled.sum()
        
        self.has_partition = True # set the partition flag to be True
    

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

        return Xas2QualityDataset(spec=red_norm, label=labels.reshape(-1,1), feature=feature_grid,
                                  energy_cut=energy_cut, verbose=verbose)

   

def get_Xas2Quality_dataloaders(dataset, batch_size, ratio=(0.7, 0.15, 0.15), unlabel_batch_size=0):
    
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
    if unlabel_batch_size != 0: 
        ds_unlabeled = Subset(dataset, dataset.index_unlabeled)
        unlabel_sampler = RandomSampler(ds_unlabeled, replacement=True, # length = len(train_loader)
                                        num_samples=unlabel_batch_size*len(train_loader)) 
        unlabel_loader = DataLoader(ds_unlabeled, batch_size=unlabel_batch_size, sampler=unlabel_sampler,
                                    num_workers=0, pin_memory=False)
        return train_loader, val_loader, test_loader, unlabel_loader

    return train_loader, val_loader, test_loader 


# test this script
if __name__ == "__main__":
    import os
    print("CWD: {:s}".format(os.getcwd()))
    data_folder = os.path.join(os.getcwd(),"large_data")
    spec = Xas2QualityDataset.from_file(spec_file=os.path.join(data_folder,"e7600-8000_grid400_spec_norm.pkl"),
                                    quality_file=os.path.join(data_folder,"e7600-8000_grid400_prediction.pkl"),
                                    energy_cut=(7600,7900))
    train_loader, test_loader, val_loader, unlabel_loader = \
        get_Xas2Quality_dataloaders(spec, batch_size=100,unlabel_batch_size=200)
    pass