#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:23:15 2019

@author: root
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import os
import sys
import random

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import feature

from skimage.filters import sobel
from skimage.morphology import watershed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable

from torchvision import transforms as tf

import h5py

from pathlib import Path
import nibabel as nib
# from sklearn import preprocessing
from skimage import transform

# from tqdm import tqdm

# from imgaug import augmenters as iaaot as plt
import pandas as pd

import sys
sys.path.insert(0, '../networks/')
from Att_Net import *
# from help_functions import *

# #function to obtain data for training/testing (validation)
# from extract_patches import get_data_training
# sys.path.insert(0, '../lib/networks/')

from preprocessing import preprocessing


#===============#set a few things====================================
save_path = 'expr/b_c_d_e/exp1_2heads/'
exp_path = 'src/'+save_path
model = Model_B_C_D_E(32,2)
model.cuda()

#====================================================================


train_img, label_img = preprocessing(exp_path)


N_subimgs = 190000
indices = list(range(N_subimgs))
#np.random.shuffle(indices)

val_size = 1/10
split = np.int_(np.floor(val_size * N_subimgs))

train_idxs = indices[split:]
val_idxs = indices[:split]


class eye_dataset(torch.utils.data.Dataset):

    def __init__(self,preprocessed_images, train=True, label=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.images = preprocessed_images
        if self.train:
            self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img = np.zeros_like(image, dtype=np.float32)
        
        img += image
        label = None
        if self.train:
            label = self.label[idx]
#             msk = np.zeros((2,48,48), dtype=np.long)
#             msk[1] = label
#             msk[0] = 1-label
            
#             msk += label
            return (img, label)
        return img

eye_dataset_train = eye_dataset(train_img[train_idxs], 
                                      train=True, 
                                      label=label_img[train_idxs])

eye_dataset_val = eye_dataset(train_img[val_idxs], 
                                      train=True, 
                                      label=label_img[val_idxs])


batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=eye_dataset_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=eye_dataset_val, 
                                           batch_size=batch_size, 
                                           shuffle=True)



# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=.1,
                             weight_decay=.000001)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                                 milestones=[400,800,900], gamma=0.1)

def accuracy(out, labels):
    total = 0.0
    predicted = torch.exp(out)
    size = predicted.shape[0]*predicted.shape[2]
    pred = torch.argmax(predicted.data, dim=1)
    total += torch.sum(pred == labels.data)
    return total.cpu().detach().numpy()/size


mean_train_losses = []
mean_val_losses = []

mean_train_acc = []
mean_val_acc = []
minLoss = 99999
maxValacc = -99999
for epoch in range(1000):
    scheduler.step()
    print('EPOCH: ',epoch+1)
#     train_losses = []
#     val_losses = []    
    train_acc = []
    val_acc = []
    
    running_loss = 0.0
    
    model.train()
    count = 0
    for images, labels in train_loader:    
#         labels = labels.squeeze()
        images = Variable(images.cuda())
        labels = labels.type(torch.LongTensor)
        labels = Variable(labels.cuda())
        
        outputs = model(images) 
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        
        train_acc.append(accuracy(outputs, labels))
        
        loss.backward()
        optimizer.step()        
        
        running_loss += loss.item()
        count +=1
    
    print('Training loss:.......', running_loss/count)
#     print('Training accuracy:...', np.mean(train_acc))
    mean_train_losses.append(running_loss/count)
        
    model.eval()
    count = 0
    val_running_loss = 0.0
    for images, labels in val_loader:
#         labels = labels.squeeze()
        images = Variable(images.cuda())
        labels = labels.type(torch.LongTensor)
        labels = Variable(labels.cuda())
                
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_acc.append(accuracy(outputs, labels))
        val_running_loss += loss.item()
        count +=1

    mean_val_loss = val_running_loss/count
    print('Validation loss:.....', mean_val_loss)
    print('')    
    print('Training accuracy:...', np.mean(train_acc))
    print('Validation accuracy..', np.mean(val_acc))
    
    mean_val_losses.append(mean_val_loss)
    
    mean_train_acc.append(np.mean(train_acc))
    
    val_acc_ = np.mean(val_acc)
    mean_val_acc.append(val_acc_)
    
   
    if mean_val_loss < minLoss:
        torch.save(model.state_dict(), save_path+'best_loss.pth' )
        print(f'NEW BEST Loss: {mean_val_loss} ........old best:{minLoss}')
        minLoss = mean_val_loss
        print('')
    
    if (epoch+1)%100 ==0:
        torch.save(model.state_dict(), save_path+'epoch_'+str(epoch+1)+'.pth')
        
        
    if val_acc_ > maxValacc:
        torch.save(model.state_dict(), save_path+'best_acc.pth' )
        print(f'NEW BEST Acc: {val_acc_} ........old best:{maxValacc}')
        maxValacc = val_acc_
    
    
    print('')


























