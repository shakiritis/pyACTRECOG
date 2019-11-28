"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


from coreLib.utils import data_input_fn,readJson

import numpy as np 
import matplotlib.pyplot as plt 


import tensorflow as tf 
#--------------------------------------------------------------------------------------------------
tf.compat.v1.enable_eager_execution()
#--------------------------------------------------------------------------------------------------
class FLAGS:
    TFRECORDS_DIR   = '/media/ansary/DriveData/HandAction/SHORT/DataSet/TFRECORD/Train/'
    IMAGE_DIM       = 256
    NB_CHANNELS     = 3
    BATCH_SIZE      = 128
    SHUFFLE_BUFFER  = 800
    MODE            = 'Train'
    NB_CLASSES      = 17
    MIN_SEQ_LEN     = 8
#--------------------------------------------------------------------------------------------------
 
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
    
if __name__ == "__main__":
    check_data()
