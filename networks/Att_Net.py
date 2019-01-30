#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:13:18 2019

@author: tony
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention


def bn_relu(chanIn, chanOut, ks = 3, stride=1, padding=1):
    return nn.Sequential(
        nn.BatchNorm2d(chanIn),
        nn.ReLU6(inplace=True),
        nn.Conv2d(chanIn, chanOut, ks, stride, padding=padding),
        
    )

class Att_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_block_1 = bn_relu(1, 1, 3,1)
        self.input_block_2 = bn_relu(1, 1, 3,1)
        self.input_1x1 = nn.Conv2d(1,32,1,1)
        
        self.down1_block_1 = bn_relu(32, 64, 3,2)
        self.down1_block_2 = bn_relu(64,64, 3,1)
        self.down1_1x1 = nn.Conv2d(32, 64, 1,2)
        
        self.down2_block_1 = bn_relu(64, 128, 3,2)
        self.down2_block_2 = bn_relu(128, 128, 3,1)
        self.down2_1x1 = nn.Conv2d(64, 128, 1,2)
        
#         self.mid = MultiHeadAttention_(128,128,32,32,4)
        self.mid = MultiHeadAttention(128,128,64,64,16)
        self.bn = nn.BatchNorm2d(128)
#         self.mid = nn.Conv2d(128,128,3,padding=1)
#         self.bn = nn.BatchNorm2d(128)
        
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2_1 = nn.Conv2d(128,64,3,1,1)
#         self.up2m = MultiHeadAttention_(64,64,12,12,4)
#         self.bn2 = nn.BatchNorm2d(64)
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.up1_1 = nn.Conv2d(64,32,3,1,1)
#         self.up1m = MultiHeadAttention_(32,32,12,12,4)
#         self.bn3 = nn.BatchNorm2d(32)
        
        self.out = nn.Conv2d(32,2,3,1,1)
       
        
        
        
    def forward(self, x):
#         print('input shape ', x.shape)
        x_top = x
        x = self.input_block_1(x)
        x = self.input_block_2(x)
        x = torch.add(x_top,x) #res1
#         print('x before 1x1 shape: ',x.shape)
        x_res1 = self.input_1x1(x)
#         print('xres1  ', x_res1.shape)
        
        x_l = self.down1_1x1(x_res1)
        x = self.down1_block_1(x_res1)
        x = self.down1_block_2(x)
        x_res2 = torch.add(x_l, x) 
        x_l = self.down2_1x1(x_res2)
        x = self.down2_block_1(x_res2)
        x = self.down2_block_2(x)
        x_res3 = torch.add(x_l,x) 
        
#       res3 shape  torch.Size([32, 128, 12, 12])
        x = self.mid(x_res3)

        x = self.bn(x)
        
        x = self.up2(x)
        x = self.up2_1(x)
        
        x = torch.add(x, x_res2)
        
        x = self.up1(x)
        x = self.up1_1(x)
        x = torch.add(x, x_res1)
        x = self.out(x)
#         print('before perm ', x.shape)
        x = x.view(x.shape[0], x.shape[1],-1)
#         x = x.permute(0,2,1)
        
#         print('out : ', x.shape)
        x =  F.log_softmax(x,dim=1)
#         print('outshape', x.shape)
        return x