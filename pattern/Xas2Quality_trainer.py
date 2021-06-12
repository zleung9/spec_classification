import torch
from torch import nn, tensor, optim
from torch.utils.data import Dataset, DataLoader

import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, data,
                 device = torch.device('cuda'), verbose=True, board=False,
                 learning_rate=1e-4, max_epochs=1000, batch_size=100):
        
        # configuration paremeters
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.tensorboard = board
        self.learning_rate = learning_rate
        
        # process input data and label data (will be replaced by dataloader)
        self.x_train, self.y_train = data['train']
        self.x_val, self.y_val = data['val']
        self.N_train = self.x_train.size()[0]
        self.N_val = self.x_val.size()[0]
        
        # define model, loss function and optimizer
        self.model = model.to(self.device)
        self.loss_BCE = nn.BCELoss(reduction='mean').to(self.device)
        self.solver: optim.Optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
        # zero_grad

    def train(self, test_set):
        
        test_norm, test_label = test_set

        if self.tensorboard:
            writer = SummaryWriter(comment="_train_all-bvs")
            print("Tensorboard is on")
        else:
            werite = None
            print("Tensorboard is off")
                
        train_loss = np.zeros(self.max_epochs)
        val_loss = np.zeros(self.max_epochs)

        if self.verbose: start = time.time() # start time
        for epoch in range(self.max_epochs):
            self.model.train() # set to training mode

            permutation = torch.randperm((self.N_train),device=self.device) # random permutation of data
            permutation_test = torch.randperm((test_norm.shape[0]),device=self.device) # 0611
            for i in range(0, self.N_train, self.batch_size):     
                self.model.zero_grad() # clear up the gradients in the model
                indices = permutation[i:i+self.batch_size] # minibatch
                y_pred = self.model(self.x_train[indices]).to(device=self.device) # predict values
                loss = self.loss_BCE(y_pred, self.y_train[indices]) # calculate loss
                loss.backward() # compute the gradients of loss in the graph
                self.solver.step() # update weights and paremeters
                
                self.model.zero_grad()
                indices_test = permutation_test[i:i+self.batch_size] # 0611
                y_pred = self.model(test_norm[indices_test]).to(device=self.device) # predict values #0611
                perc30, perc70 = torch.quantile(y_pred, tensor([0.3, 0.7]).to(device=self.device))
                
                i_invalid30 = (y_pred > perc30) & (y_pred < 0.3)
                invalid_pred30 = y_pred[i_invalid30]
                loss30 = self.loss_BCE(invalid_pred30, torch.full_like(invalid_pred30, fill_value=0.3))
                
                i_invalid70 = (y_pred < perc70) & (y_pred > 0.7)
                invalid_pred70 = y_pred[i_invalid70]
                loss70 = self.loss_BCE(invalid_pred70, torch.full_like(invalid_pred70, fill_value=0.3))
                
                loss_center = loss30 + loss70
                loss_center = loss_center * 0.5 # avoid distorting the good result
                loss_center.backward()
                self.solver.step() # update weights and paremeters

            self.model.eval() # set to evaluation mode
            train_loss[epoch] = self.loss_BCE(self.model(self.x_train),self.y_train) # average loss
            val_loss[epoch] = self.loss_BCE(self.model(self.x_val),self.y_val)
           
            try:
                writer.add_scalar("Loss/train", train_loss[epoch], epoch) 
                writer.add_scalar("Loss/val", val_loss[epoch], epoch)
            except:
                pass
            if self.verbose and epoch%100==99:
                print('Epoch {:d}/{:d} | loss: {:.3f}/{:.3f}'
                      .format(epoch+1, self.max_epochs, train_loss[epoch],val_loss[epoch]))    
        try: 
            writer.flush()
        except:
            pass
        if self.verbose:
            print("Total time: {:.1f} min.".format((time.time()-start)/60))  
        
        return train_loss, val_loss

