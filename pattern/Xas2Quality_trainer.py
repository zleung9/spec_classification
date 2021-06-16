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
    def __init__(self, model, train_loader, test_loader, val_loader, loss_constrain_test_loader=None, 
                 device = torch.device('cuda'), verbose=True, ts_board=False,
                 learning_rate=1e-4, max_epochs=1000, batch_size=100):
        
        # configuration paremeters
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.tensorboard = ts_board
        self.learning_rate = learning_rate
        
        # process input data and label data (will be replaced by dataloader)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # this dataloader is used for pulling the extreme probability: 0->0.3, 1->0.7, see "def train(self)"
        self.loss_constrain_loader = loss_constrain_test_loader

        # define model, loss function and optimizer
        self.model = model.to(self.device)
        self.loss_BCE = nn.BCELoss(reduction='mean').to(self.device)
        self.solver: optim.Optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
        # zero_grad

    def train(self):

        if self.tensorboard:
            writer = SummaryWriter(comment="_train_all-bvs")
            print("Tensorboard is on")
        else:
            writer = None
            print("Tensorboard is off")
                
        self.train_loss = np.zeros(self.max_epochs)
        self.val_loss = np.zeros(self.max_epochs)

        if self.verbose: start = time.time() # start time
        for epoch in range(self.max_epochs):
            self.model.train() # set to training mode


            loss_train_list = []
            #train_loader can only be exhausted if its length < that of loss_constrain_loader
            for trn, tst in zip(self.train_loader, self.loss_constrain_loader):
                spec_in, label_in = trn
                spec_in = spec_in.to(self.device)
                label_in = label_in.to(self.device)

                self.model.zero_grad()
                label_pred = self.model(spec_in)
                loss = self.loss_BCE(label_pred, label_in)
                loss_train_list.append(loss) # list of training loss for the current epoch
                loss.backward()
                self.solver.step()

                # this part tries to minimize loss w.r.t. 0.3 & 0.7 so predictions are pulled to center
                spec_tst, label_tst = tst
                spec_tst = spec_tst.to(self.device)
                label_tst = label_tst.to(self.device)
                self.model.zero_grad()
                y_pred = self.model(spec_tst).to(device=self.device)
                # only predictions that fall in [perc30, perc70] (not-so-sure predictions) are affected.
                perc30, perc70 = torch.quantile(y_pred, tensor([0.3,0.7]).to(device=self.device))
                invalid_pred30 = y_pred[(y_pred>perc30) & (y_pred<0.3)]
                loss_perc30 = self.loss_BCE(invalid_pred30, torch.full_like(invalid_pred30, fill_value=0.30))
                invalid_pred70 = y_pred[(y_red<perc70) & (y_pred>0.7)]
                loss_perc70 = self.loss_BCE(invalid_pred70, torch.full_like(invalid_pred70, full_value=0.70))
                loss_center = (loss_perc30 + loss_perc70) * 0.5 # avoid distorting the good result too much
                loss_center.backward()
                self.solver.step()

            # caculate the loss for validition set.
            self.model.eval()
            loss_val_list = []
            for spec_in, label_in in self.val_loader:
                spec_in = spec_in.to(self.device)
                label_in = label_in.to(self.device)
                label_pred = self.model(spec_in)
                loss = self.loss_BCE(label_pred, label_in)
                loss_val_list.append(loss) # list of validation loss for the current epoch
                
            self.train_loss[epoch] = np.mean(loss_train_list)
            self.val_loss[epoch] = np.mean(loss_val_list)

            try:
                writer.add_scalar("Loss/train", self.train_loss[epoch], epoch) 
                writer.add_scalar("Loss/val", self.val_loss[epoch], epoch)
            except:
                pass
            if self.verbose and epoch%100==99:
                print('Epoch {:d}/{:d} | loss: {:.3f}/{:.3f}'
                      .format(epoch+1, self.max_epochs, train_loss[epoch],val_loss[epoch]))    
        try: 
            writer.flush() # flush the writer opject
        except:
            pass
        
        if self.verbose:
            print("Total time: {:.1f} min.".format((time.time()-start)/60))  


    def test(self):
        pass