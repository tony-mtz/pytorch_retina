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
        nn.BatchNorm2d(chanIn,momentum=.997),
        nn.ReLU6(inplace=True),
        nn.Conv2d(chanIn, chanOut, ks, stride, padding=padding),
        nn.BatchNorm2d(chanOut,momentum=.997),
        nn.ReLU6(inplace=True),
        nn.Conv2d(chanOut, chanOut, ks, stride=1, padding=1)
        
    )
    

class Att_Net(nn.Module):
    def __init__(self, input_size):
        self.chn = input_size
        super().__init__()
        self.input_conv1 = nn.Conv2d(1,self.chn,3,1,1)
        self.block1 = bn_relu(self.chn, self.chn,3,1,1)
        
        self.block2 = bn_relu(self.chn, self.chn*2,3,2) #down
        self.block2_res = nn.Conv2d(self.chn, self.chn*2,1,2, padding=0)
        
        self.block3 = bn_relu(self.chn*2, self.chn*4, 3,2)
        self.block3_res = nn.Conv2d(self.chn*2, self.chn*4,1,2, padding=0)
        
        self.mid = MultiHeadAttention(self.chn*4,self.chn*4,32,32,8)
        self.bn = nn.BatchNorm2d(self.chn*4,momentum=.997)
        
        
        self.up2 = nn.ConvTranspose2d(self.chn*4, self.chn*2, 3,2, padding=1, output_padding=1)
        self.up2_1 = bn_relu(self.chn*2,self.chn*2, 3,1,1)

        self.up1 = nn.ConvTranspose2d(self.chn*2, self.chn, 3,2, padding=1, output_padding=1)
        self.up1_1 = bn_relu(self.chn,self.chn,3,1,1)
#         self.up1m = MultiHeadAttention_(32,32,12,12,4)
#         self.bn3 = nn.BatchNorm2d(32)     
        self.drop = nn.Dropout2d(.5)
        self.out = bn_relu(self.chn, self.chn,3,1,1)
        self.out_ = nn.Conv2d(self.chn,2,3,1,1)      
        
        
    def forward(self, x):
#        print('input shape ', x.shape)
        x_top = self.input_conv1(x)
        x = self.block1(x_top)
        res1 = torch.add(x_top,x) #res1
#        print('res1 shape: ', res1.shape)
        
        x_top = self.block2_res(res1)
#        print('x top shape: ', x_top.shape)
        x = self.block2(res1)
#        print('x top shape: ', x.shape)
        x_res2 = torch.add(x_top, x)#res2
#        print('res2 shape: ', x_res2.shape)
        
        x_top = self.block3_res(x_res2)
        x = self.block3(x_res2)
        x = torch.add(x_top, x) 
        
#       res3 shape  torch.Size([32, 128, 12, 12])
        x = self.mid(x)
        x = self.bn(x)
#        print('out mid: ', x.shape)
        
        x = self.up2(x)    
#        print('out transpose ', x.shape)
        x = torch.add(x, x_res2)
#        print('x after add ', x.shape)
        x_l = self.up2_1(x)
        x_out = torch.add(x_l,x)
        
        
        
        x = self.up1(x_out)
        x = torch.add(x, res1)
        x_l = self.up1_1(x)
        x_out = torch.add(x_l,x)
        
        x = self.drop(x_out)
        x = self.out(x)
        x = self.out_(x)
#         print('before perm ', x.shape)
        x = x.view(x.shape[0], x.shape[1],-1)
#         x = x.permute(0,2,1)
        
#         print('out : ', x.shape)
        x =  F.log_softmax(x,dim=1)
#         print('outshape', x.shape)
        return x