#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


from coreLib.utils import data_input_fn,readJson,LOG_INFO
from coreLib.model import convNet3D,LRCN
import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling3D, Conv3D,Dense,Flatten,TimeDistributed,LSTM,BatchNormalization,Activation,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import tensorflow as tf 
#--------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Hand Action Recognition: debug script ')
parser.add_argument("tfrecord_dir", help='/path/to/tfrecord/')
parser.add_argument("exec_flag", help='Check Data or Train a Model. Available flags: CHECK,ConvNet3D,LRCN,DUMMY')
args = parser.parse_args()
if args.exec_flag=='CHECK':
    tf.compat.v1.enable_eager_execution()
#--------------------------------------------------------------------------------------------------
class FLAGS:
    TFRECORDS_DIR   = args.tfrecord_dir
    IMAGE_DIM       = 64
    NB_CHANNELS     = 3
    BATCH_SIZE      = 2
    SHUFFLE_BUFFER  = 100
    MODE            = 'Train'
    NB_CLASSES      = 17
    MIN_SEQ_LEN     = 6
#--------------------------------------------------------------------------------------------------
NB_TRAIN_DATA       = 49920
NB_EVAL_DATA        = 3456
NB_TOTAL_DATA       = NB_TRAIN_DATA + NB_EVAL_DATA 
STEPS_PER_EPOCH     =  NB_TOTAL_DATA //FLAGS.BATCH_SIZE 
VALIDATION_STEPS    =  NB_EVAL_DATA //FLAGS.BATCH_SIZE 
N_EPOCHS            = 2
#--------------------------------------------------------------------------------------------------
def train_in_fn():
    return data_input_fn(FLAGS)
    
def eval_in_fn():
    FLAGS.MODE='Eval'
    return data_input_fn(FLAGS)
#--------------------------------------------------------------------------------------------------
def check_data():
    dataset=data_input_fn(FLAGS)
    for X, y in dataset:
        for feat in X[0]:
            plt.imshow(np.squeeze(feat))
            plt.show()
            print(feat[0][0])
#--------------------------------------------------------------------------------------------------
def dummy_model(FLAGS):
    IN=Input(shape=(FLAGS.MIN_SEQ_LEN,FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
    X=Conv3D(8, (3,3,3), activation='relu',padding='same')(IN)
    X=Flatten()(X)
    X=Dense(FLAGS.NB_CLASSES, activation='softmax')(X)
    return Model(inputs=IN,outputs=X)

def train_debug(args):
    if args.exec_flag=='LRCN':
        model=LRCN(seq_len=FLAGS.MIN_SEQ_LEN,
                        img_dim=FLAGS.IMAGE_DIM,
                        nb_channels=FLAGS.NB_CHANNELS,
                        nb_classes=FLAGS.NB_CLASSES)
    elif args.exec_flag=='ConvNet3D':
        model=convNet3D(seq_len=FLAGS.MIN_SEQ_LEN,
                        img_dim=FLAGS.IMAGE_DIM,
                        nb_channels=FLAGS.NB_CHANNELS,
                        nb_classes=FLAGS.NB_CLASSES)
    elif args.exec_flag=='DUMMY':
        model=dummy_model(FLAGS)
    else:
        raise ValueError('Use Proper Model Name')
    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    model.summary()
    model.fit(train_in_fn(),
            epochs= N_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH, 
            validation_data=eval_in_fn(),
            validation_steps=VALIDATION_STEPS,
            verbose=1)

def main(args):
    if args.exec_flag=='CHECK':
        check_data()
    else:
        train_debug(args)
if __name__ == "__main__":
    main(args)
    