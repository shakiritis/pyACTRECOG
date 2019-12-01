#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import os
import numpy as np 
import matplotlib.pyplot as plt
import time

from coreLib.model import convNet3D
from coreLib.utils import LOG_INFO,preprocess_sequence,readJson,dump_data

from sklearn import metrics
from tensorflow.keras.models import load_model

from progressbar import ProgressBar

#--------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Hand Action Recognition Testing')
parser.add_argument("model_name", help='name of the model. Available:ConvNet3D')
parser.add_argument("model_path", help='Path for model weights')
parser.add_argument("json_path", help='Path for Test Sequence json')
args = parser.parse_args()
#--------------------------------------------------------------------------------------------------------------------------------------------------
class FLAGS:
    IMAGE_DIM       = 32
    NB_CHANNELS     = 3
    NB_CLASSES      = 17
    MIN_SEQ_LEN     = 6

CLASS_LIST=['ui-screenTap', 
            'asl-you', 
            'asl-blue', 
            'ui-doubleclick', 
            'asl-j', 
            'asl-green', 
            'asl-z', 
            'ui-keyTap', 
            'asl-scissors', 
            'asl-bathroom', 
            'ui-swipe', 
            'asl-where', 
            'asl-yellow', 
            'ui-click', 
            'asl-milk', 
            'ui-circle', 
            'no-action']
#--------------------------------------------------------------------------------------------------------------------------------------------------
def main(args):
    _pbar=ProgressBar()
    GROUNT_TRUTHS=[]
    PREDICTIONS=[]
    VID_DICT=[]
    if args.model_name=='ConvNet3D':
        model=convNet3D(seq_len=FLAGS.MIN_SEQ_LEN,
                        img_dim=FLAGS.IMAGE_DIM,
                        nb_channels=FLAGS.NB_CHANNELS,
                        nb_classes=FLAGS.NB_CLASSES)
    else:
        raise ValueError('CHECK Proper Model Name')

    model.summary()
    model.load_weights(args.model_path)
    LOG_INFO('Generating Predictions')
    sequences=readJson(args.json_path)
    for sequence in _pbar(sequences):
        X,Y_TRUTH=preprocess_sequence(sequence,CLASS_LIST,FLAGS.IMAGE_DIM)   
        GROUNT_TRUTHS.append(Y_TRUTH)
        Y_PRED=np.argmax(model.predict(np.expand_dims(X,axis=0)))
        PREDICTIONS.append(Y_PRED)

        img_dir=sequence['path']
        id_start=int(sequence['start'])
        id_stop=int(sequence['stop'])
        class_id=sequence['class']    
        pred_id=CLASS_LIST[Y_PRED]
        image_path=os.path.join(img_dir,'color_{}.png'.format(id_start))
        IMG_DICT={'path':image_path,
                  'truth':class_id,
                  'prediction':pred_id}

        VID_DICT.append(IMG_DICT)

    DS_DIR=os.path.dirname(args.json_path)
    RES_JSON=os.path.join(DS_DIR,'video_sec_{}.json'.format(args.model_name))
    dump_data(RES_JSON,VID_DICT)
    prediction_accuracy = 100* metrics.f1_score(GROUNT_TRUTHS,PREDICTIONS, average = 'micro')	   
    LOG_INFO('Test data Prediction Accuracy [F1 accuracy]: {}'.format(prediction_accuracy))

if __name__ == "__main__":
    main(args)
