#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:07:06 2019

@author: tony
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, '../lib/')
from help_functions import *
from extract_patches import get_data_training

#function to obtain data for training/testing (validation)
def preprocessing(exp_path):
    from extract_patches import get_data_training
    sys.path.insert(0, '../lib/networks/')
#    from fully_conv import get_fully_conv_unet as unet
    
    # [data paths]
    path_local =  '../DRIVE_datasets_training_testing/'
    train_imgs_original = 'DRIVE_dataset_imgs_train.hdf5'
    train_groundTruth = 'DRIVE_dataset_groundTruth_train.hdf5'
    train_border_masks = 'DRIVE_dataset_borderMasks_train.hdf5'
    test_imgs_original = 'DRIVE_dataset_imgs_test.hdf5'
    test_groundTruth = 'DRIVE_dataset_groundTruth_test.hdf5'
    test_border_masks = 'DRIVE_dataset_borderMasks_test.hdf5'
    
    
    
    # [experiment name]
    name_experiment = exp_path
    
    
    # [data attributes]
    #Dimensions of the patches extracted from the full images
    patch_height = 48
    patch_width = 48
    
    
    # [training settings]
    #number of total patches:
    N_subimgs = 190000
    # 190000
    #if patches are extracted only inside the field of view:
    inside_FOV = True
    #Number of training epochs
    N_epochs = 2
    batch_size =64
    #if running with nohup
    nohup = True
    zca = False
    
    
    # [testing settings]
    #Choose the model to test: best==epoch with min loss, last==last epoch
    best_last ='best'
    #number of full images for the test (max 20)
    full_images_to_test = 20
    #How many original-groundTruth-prediction images are visualized in each image
    N_group_visual = 1
    #Compute average in the prediction, improve results but require more patches to be predicted
    average_mode = True
    #Only if average_mode==True. Stride for patch extraction, lower value require more patches to be predicted
    stride_height = 5
    stride_width = 5
    #if running with nohup
    nohup = False
    
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original = path_local + train_imgs_original,
        DRIVE_train_groudTruth = path_local + train_groundTruth,  #masks
        patch_height = int(patch_height),
        patch_width = int(patch_width),
        N_subimgs = int(N_subimgs),
        inside_FOV = inside_FOV #select the patches only inside the FOV  (default == True)
    )
    
    N_sample = min(patches_imgs_train.shape[0],40)
    visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'../'+name_experiment+'/'+"sample_input_imgs")#.show()
    visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'../'+name_experiment+'/'+"sample_input_masks")#.show()
    
    n_ch = patches_imgs_train.shape[1]
    patch_height = patches_imgs_train.shape[2]
    patch_width = patches_imgs_train.shape[3]
    print(patches_imgs_train.shape)
    print(patch_height, patch_width, n_ch)
    
    print('......DONE......')
    
    patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
    
    return patches_imgs_train, patches_masks_train
