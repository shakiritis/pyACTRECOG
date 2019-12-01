"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


from coreLib.utils import data_input_fn,readJson
from coreLib.model import convNet3D
import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling3D, Conv3D,Dense,Flatten,TimeDistributed,LSTM,BatchNormalization,Activation,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

DIR='/media/ansary/DriveData/HandAction/DataSet/HACTION/'
import tensorflow as tf 
#--------------------------------------------------------------------------------------------------
# tf.compat.v1.enable_eager_execution()
#--------------------------------------------------------------------------------------------------
class FLAGS:
    TFRECORDS_DIR   = DIR
    IMAGE_DIM       = 32
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
def train_in_fn():
    dataset=data_input_fn(FLAGS)
    iterator = dataset.make_one_shot_iterator()
    X, y = iterator.get_next()
    while True:
        with tf.Session() as sess:
            feats=X.eval()
            label=y.eval()
        yield feats,label
    
def eval_in_fn():
    FLAGS.MODE='Eval'
    dataset=data_input_fn(FLAGS)
    iterator = dataset.make_one_shot_iterator()
    X, y = iterator.get_next()
    while True:
        with tf.Session() as sess:
            feats=X.eval()
            label=y.eval()
        yield feats,label
        

def check_data():
    dataset=data_input_fn(FLAGS)
    iterator = dataset.make_one_shot_iterator()
    X, y = iterator.get_next()
    while True:
        with tf.Session() as sess:
            feats=X.eval()
            label=y.eval()
        for feat in feats[0]:
            plt.imshow(np.squeeze(feat))
            plt.show()

            print(feat[0][0])
def dummy_model(FLAGS):
    IN=Input(shape=(FLAGS.MIN_SEQ_LEN,FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
    X=Conv3D(8, (3,3,3), activation='relu',padding='same')(IN)
    X=Flatten()(X)
    X=Dense(FLAGS.NB_CLASSES, activation='softmax')(X)
    return Model(inputs=IN,outputs=X)

def train_debug():
    #model=dummy_model(FLAGS)
    model=convNet3D(seq_len=FLAGS.MIN_SEQ_LEN,
                    img_dim=FLAGS.IMAGE_DIM,
                    nb_channels=FLAGS.NB_CHANNELS,
                    nb_classes=FLAGS.NB_CLASSES)
    model.summary()
    #plot_model(model,to_file='dummy.png',show_layer_names=True,show_shapes=True)
    model.compile(loss='categorical_crossentropy',optimizer=Adam())
    model.fit_generator(train_in_fn(),
            epochs= N_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH, 
            validation_data=eval_in_fn(),
            validation_steps=VALIDATION_STEPS,
            verbose=1)

if __name__ == "__main__":
    #check_data()
    train_debug()
    