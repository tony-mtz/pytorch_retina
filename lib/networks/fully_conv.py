#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:20:39 2019

@author: tony
"""

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import  layers
import numpy as np

import sys
sys.path.insert(0, '../')
from attention import *

class Fully_Conv(tf.keras.Model):

    def __init__(self):
        super(Fully_Conv, self).__init__()
        
        chan =  'channels_first'
        #channels last is the default
        # Define your layers here.
        self.conv1_1 = layers.Conv2D(32,3, activation='relu', padding='same', data_format=chan)
        self.drop1 = layers.Dropout(.2)
        self.conv1_2 = layers.Conv2D(32,3, activation='relu', strides=2, padding='same', data_format=chan)
        
        self.conv2_1 = layers.Conv2D(64,3, activation='relu', padding='same', data_format=chan)
        self.drop2 = layers.Dropout(.2)
        self.conv2_2 = layers.Conv2D(64,3, activation='relu', strides=2, padding='same', data_format=chan)
       
        #bottom
        self.conv3_1 = layers.Conv2D(128,3, activation='relu', padding='same', data_format=chan)
        self.drop3 = layers.Dropout(.2)
        self.conv3_2 = layers.Conv2D(128,3, activation='relu', padding='same', data_format=chan)
        self.pool3 = layers.MaxPool2D()
        
        self.up1 = layers.UpSampling2D(size=(2,2), data_format=chan)
        
        self.conv4_1 = layers.Conv2D(64,3, activation='relu', padding='same', data_format=chan)
        self.drop4 = layers.Dropout(.2)
        self.conv4_2 = layers.Conv2D(64,3, activation='relu', padding='same', data_format=chan)
        
        self.up2 = layers.UpSampling2D(size=(2,2), data_format=chan)
        
        self.conv5_1 = layers.Conv2D(32,3, activation='relu', padding='same', data_format=chan)
        self.drop5 = layers.Dropout(.2)
        self.conv5_2 = layers.Conv2D(32,3, activation='relu', padding='same', data_format=chan)
    
        
        self.conv6_1 = layers.Conv2D(2,1, activation='relu', padding='same',data_format=chan)
#         self.drop6 = layers.Dropout(.2)
#         self.conv6_2 = layers.Conv2D(64,3, activation='relu', padding='same')
        
        self.act = layers.Dense(2, activation='softmax')
    
        
    def call(self, input):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
#         inputs = input.get_shape[1]*input.get_shape[2]
        x_1 = self.conv1_1(input)
        print('x_1, ',x_1.get_shape()) #x_1,  (?, 48, 48, 32)
        x = self.drop1(x_1)
        x = self.conv1_2(x)
        
        x_2 = self.conv2_1(x)
        x = self.drop2(x_2)
        x = self.conv2_2(x)
        
        x_3 = self.conv3_1(x)
        x = self.drop3(x_3)
        x = self.conv3_2(x)
        print('after conv3-2', x.shape)
        
        x = self.up1(x)
        print('aster up1 ', x.shape)
        x = layers.Concatenate(axis=1)([x_2,x])
        x = self.conv4_1(x)
        x = self.drop4(x)
        x = self.conv4_2(x)
        
        x = self.up2(x)
        print('aster up2 x ', x.shape)
        x = layers.Concatenate(axis=1)([x_1,x])
        x = self.conv5_1(x)
        x = self.drop5(x)
        x = self.conv5_2(x)
        
        x = self.conv6_1(x)
        
#       conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
#       conv6 = core.Permute((2,1))(conv6)


        x = layers.Reshape((2,48*48))(x)
        print('reshape : ', x.shape)
        x = layers.Permute((2,1))(x)
        x = self.act(x)
        print('perm : ', x.shape)
        return x
    
    
def get_fully_conv_unet(n_ch,patch_height, patch_width):
    chan = 'channels_first'
    inputs = layers.Input(shape=(n_ch,patch_height, patch_width))
    
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format=chan)(inputs)
    conv1 = layers.Dropout(0.2)(conv1)
    conv1_strided = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=2,data_format=chan)(conv1)
    
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format=chan)(conv1_strided)
    conv2 = layers.Dropout(0.2)(conv2)
    conv2_strided = layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=2,data_format=chan)(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format=chan)(conv2_strided)
    conv3 = layers.Dropout(0.2)(conv3)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format=chan)(conv3)

    up1 = layers.UpSampling2D(size=(2, 2), data_format=chan)(conv3)
    up1 = layers.Concatenate(axis=1)([conv2, up1])
    
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format=chan)(up1)
    conv4 = layers.Dropout(0.2)(conv4)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format=chan)(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2), data_format=chan)(conv4)
    up2 = layers.Concatenate(axis=1)([conv1, up2])
    
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format=chan)(up2)
    conv5 = layers.Dropout(0.2)(conv5)
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format=chan)(conv5)
    
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same',data_format=chan)(conv5)
    conv6 = layers.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.Permute((2,1))(conv6)
    
    conv7 = layers.Activation('softmax')(conv6)

    model = keras.Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    adam = tf.keras.optimizers.Adam(lr=0.001, decay=.01)#, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model




def attend(n_ch,patch_height, patch_width):
    chan = 'channels_first'
    inputs = layers.Input(shape=(n_ch,patch_height, patch_width))
    
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format=chan)(inputs)
    conv1 = layers.Dropout(0.2)(conv1)
    conv1_strided = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=2,data_format=chan)(conv1)
    
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format=chan)(conv1_strided)
    conv2 = layers.Dropout(0.2)(conv2)
    conv2_strided = layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=2,data_format=chan)(conv2)
    
#     conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format=chan)(conv2_strided)
#     conv3 = layers.Dropout(0.2)(conv3)
#     conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format=chan)(conv3)
    print(conv2_strided)
    roll = tf.transpose(conv2_strided, [0,2,3,1])
    att = keras.layers.Lambda(lambda x: multihead_attention_2d(inputs=roll, total_key_filters=64,
                                                               total_value_filters=64, output_filters=128, 
                                                               num_heads=8, training=True,
                                                               layer_type='SAME'))(roll) 
    roll = tf.transpose(att, [0,3,1,2])
    #up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
#     up2 = keras.layers.Concatenate(axis=3)([conv1, roll])
    
    
    up1 = layers.UpSampling2D(size=(2, 2), data_format=chan)(roll)
    up1 = layers.Concatenate(axis=1)([conv2, up1])
    print('shape up1', up1.shape)
    
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format=chan)(up1)
    conv4 = layers.Dropout(0.2)(conv4)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format=chan)(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2), data_format=chan)(conv4)
    print('shape up2', up2.shape)
    up2 = layers.Concatenate(axis=1)([conv1, up2])
    
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format=chan)(up2)
    conv5 = layers.Dropout(0.2)(conv5)
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format=chan)(conv5)
    
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same',data_format=chan)(conv5)
    conv6 = layers.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.Permute((2,1))(conv6)
    
    conv7 = layers.Activation('softmax')(conv6)
    print(conv7)
    model = keras.Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    adam = tf.keras.optimizers.Adam(lr=0.001, decay=.01)#, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
























