"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


from coreLib.utils import data_input_fn,readJson

import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling3D, Conv3D,Dense,Flatten,TimeDistributed,LSTM,BatchNormalization,Activation,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow as tf 
#--------------------------------------------------------------------------------------------------
tf.compat.v1.enable_eager_execution()
#--------------------------------------------------------------------------------------------------
class FLAGS:
    TFRECORDS_DIR   = '/media/ansary/DriveData/HandAction/SHORT/DataSet/TFRECORD/Train/'
    IMAGE_DIM       = 256
    NB_CHANNELS     = 3
    BATCH_SIZE      = 2
    SHUFFLE_BUFFER  = 800
    MODE            = 'Train'
    NB_CLASSES      = 17
    MIN_SEQ_LEN     = 8
#--------------------------------------------------------------------------------------------------
NB_TOTAL_DATA       = 20
N_EPOCHS            = 2
STEPS_PER_EPOCH     = NB_TOTAL_DATA //FLAGS.BATCH_SIZE 
def train_in_fn():
    return data_input_fn(FLAGS) 

def check_data():
    dataset=data_input_fn(FLAGS)
    iterator = dataset.make_one_shot_iterator()
    X, y = iterator.get_next()
    feats=X[0]
    label=y[0]
    print(label)
    for feat in feats:
        plt.imshow(feat)
        plt.show()

def dummy_model(FLAGS):
    IN=Input(shape=(FLAGS.MIN_SEQ_LEN,FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
    X=Conv3D(8, (3,3,3), activation='relu',padding='same')(IN)
    X=Flatten()(X)
    X=Dense(FLAGS.NB_CLASSES, activation='softmax')(X)
    return Model(inputs=IN,outputs=X)

def train_debug():
    model=dummy_model(FLAGS)
    model.summary()
    plot_model(model,to_file='dummy.png',show_layer_names=True,show_shapes=True)
    model.compile(loss='categorical_crossentropy',optimizer=Adam())
    model.fit(train_in_fn(),epochs= N_EPOCHS,steps_per_epoch=10,verbose=1)

if __name__ == "__main__":
    #check_data()
    train_debug()
    