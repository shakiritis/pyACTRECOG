# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv3D,MaxPooling3D,Activation,ZeroPadding3D
from tensorflow.keras.layers import Dropout,Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D,Input,Concatenate,BatchNormalization,Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

import os
#--------------------------------------------------------------------------------------
def convNet3D(seq_len=6,img_dim=64,nb_channels=3,nb_classes=17):
    in_shape=(seq_len,img_dim,img_dim,nb_channels)
    feature_spec=[128,256,512]
    IN=Input(shape=in_shape)
    # first conv group
    X=Conv3D(64, (3, 3, 3), padding='same', name='CONV3D_1')(IN)
    X=MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='POOL3D_1')(X)
    X=Activation('relu',name='ACTIVATION_1')(X)
    X=BatchNormalization(name='BATCH_NORM_1')(X)
    # conv layer gropus
    for n,nb_filter in enumerate(feature_spec):
        X=Conv3D(nb_filter, (3, 3, 3), padding='same', name='CONV3D_{}'.format(n+2))(X)
        X=MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='POOL3D_{}'.format(n+2))(X)
        X=Activation('relu',name='ACTIVATION_{}'.format(n+2))(X)
        X=BatchNormalization(name='BATCH_NORM_{}'.format(n+2))(X)
    # FC layers group
    X=Flatten(name='FLATTEN')(X)
    X=Dense(4096, activation='relu', name='DENSE_1')(X)
    X=Dense(128, activation='relu', name='DENSE_2')(X)
    X=Dense(nb_classes, activation='softmax',name='DENSE_CLASS')(X)
    return Model(inputs=IN,outputs=X)
#--------------------------------------------------------------------------------------
if __name__=='__main__':
    info_path='/media/ansary/DriveData/HandAction/pyACTRECOG/info/'
    model=convNet3D()
    model.summary()
    plot_model(model,to_file=os.path.join(info_path,'convNet3D.png'),show_layer_names=True,show_shapes=True)
    