# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv3D,MaxPooling3D,Activation,ZeroPadding3D,TimeDistributed,Conv2D,MaxPooling2D
from tensorflow.keras.layers import Dropout,Dense,Flatten,GlobalAveragePooling2D,LSTM
from tensorflow.keras.layers import AveragePooling2D,Input,Concatenate,BatchNormalization,Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

import os
#--------------------------------------------------------------------------------------
def convNet3D(seq_len=6,img_dim=64,nb_channels=3,nb_classes=17,drop_out=0.2,weight_decay=1e-4):
    in_shape=(seq_len,img_dim,img_dim,nb_channels)
    feature_spec=[128,256,512,512]
    IN=Input(shape=in_shape)
    # 1st layer group	    
    X=Conv3D(64,(3, 3, 3), activation='relu',padding='same', name='INITIAL_CONV3D',kernel_regularizer=l2(weight_decay))(IN)
    X=MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='INITIAL_POOL3D')(X)
    X=Activation('relu')(X)
    X=BatchNormalization()(X)	    
    # conv layer gropus	    
    for n,nb_filter in enumerate(feature_spec):	   
        X=Conv3D(nb_filter, (3, 3, 3), activation='relu',padding='same',kernel_regularizer=l2(weight_decay), name='CONV3D_{}_C1'.format(n+1))(X)
        X=Conv3D(nb_filter, (3, 3, 3), activation='relu',padding='same',kernel_regularizer=l2(weight_decay), name='CONV3D_{}_c2'.format(n+1))(X)
        X=MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='POOL3D_{}'.format(n+1))(X)
        X=Activation('relu')(X)
        X=BatchNormalization()(X)	           
    # FC layers group	   
    X=Flatten(name='FLATTEN')(X)	    
    X=Dense(4096, activation='relu', name='DENSE_1')(X)
    if drop_out:
            X=Dropout(drop_out,name='DROP_OUT_D-1')(X)	    
    X=Dense(4096, activation='relu', name='DENSE_2')(X)
    if drop_out:
            X=Dropout(drop_out,name='DROP_OUT_D-2')(X)	    
    X=Dense(nb_classes, activation='softmax',name='DENSE_CLASS')(X)
    return Model(inputs=IN,outputs=X)

def LRCN(seq_len=6,img_dim=64,nb_channels=3,nb_classes=17,drop_out=0.2,weight_decay=1e-4):
    in_shape=(seq_len,img_dim,img_dim,nb_channels)
    feature_spec=[256,512]
    layer_specs=[256,128,64]
    IN=Input(shape=in_shape)
    # first (non-default) block
    X=TimeDistributed(Conv2D(128, (3, 3), strides=(2, 2), padding='same',kernel_regularizer=l2(weight_decay)))(IN)
    X=TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(X)
    X=TimeDistributed(Activation('relu'))(X)
    X=BatchNormalization()(X)
    # conv layer gropus
    for nb_filter in feature_spec:
        X=TimeDistributed(Conv2D(nb_filter, (3, 3), strides=(2, 2), padding='same',kernel_regularizer=l2(weight_decay)))(X)
        X=TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(X)
        X=TimeDistributed(Activation('relu'))(X)
        X=BatchNormalization()(X)
        
    # FC layers group
    X=TimeDistributed(Flatten())(X)
    for i in range(len(layer_specs)):
        X=LSTM(layer_specs[i],return_sequences=True,activation='tanh')(X)
    # last layers
    X=LSTM(32,return_sequences=False,activation='tanh')(X)
    X=Dense(28, activation='relu', name='DENSE_1')(X)
    if drop_out:
            X=Dropout(drop_out,name='DROP_OUT_D-1')(X)	    
    X=Dense(24, activation='relu', name='DENSE_2')(X)
    if drop_out:
            X=Dropout(drop_out,name='DROP_OUT_D-2')(X)	    
    X=Dense(nb_classes, activation='softmax',name='DENSE_CLASS')(X)
    return Model(inputs=IN,outputs=X)

#--------------------------------------------------------------------------------------
if __name__=='__main__':
    info_path='/media/ansary/DriveData/HandAction/pyACTRECOG/info/'
    
    model=convNet3D()
    model.summary()
    plot_model(model,to_file=os.path.join(info_path,'convNet3D.png'),show_layer_names=True,show_shapes=True)
    
    model=LRCN()
    model.summary()
    plot_model(model,to_file=os.path.join(info_path,'LRCN.png'),show_layer_names=True,show_shapes=True)
    