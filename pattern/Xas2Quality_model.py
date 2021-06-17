import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Xas2QualityFCN(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.4):
        super(Xas2QualityFCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

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
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(16, affine=False),
            nn.Linear(16,output_size),
            nn.Sigmoid()
        )   

    def forward(self, spec_in):
        assert len(spec_in.shape) == 2 # 2D matrix of shape [N,L]
        assert spec_in.shape[1] == self.input_size 
        return self.main(spec_in)

    def reset_weights(self, verbose=False):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in self.main.children():
            if hasattr(layer, 'reset_parameters'):
                if verbose: print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()


class Xas2QualityCNN(nn.Module):
    def __init__(self, output_size, dropout_rate=0.4):
        super(Xas2QualityCNN, self).__init__()
        self.output_size = output_size

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
        if len(spec_in.shape) == 2: # if input has shape [N, L], add channel dimension
            spec_in = spec_in[:,np.newaxis,:] 
        assert len(spec_in.shape) == 3 # make sure
        return self.main(spec_in)
    
