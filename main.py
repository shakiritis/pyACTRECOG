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

from coreLib.utils import LOG_INFO,DataSet,readJson,sequencesToRecord
#--------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Hand Action Recognition Preprocessing')
parser.add_argument("src_path", help='Source Data Folder')
parser.add_argument("dest_path", help='Destination Data Folder')
args = parser.parse_args()
#--------------------------------------------------------------------------------------------------------------------------------------------------
class STATS:
    IMAGE_DIM   =   32
    NB_CHANNELS =   1
    BATCH_SIZE  =   128
    FILE_LEN    =   512
#--------------------------------------------------------------------------------------------------------------------------------------------------
def createDataset(args,STATS,mode):
    DS=DataSet(args.src_path,args.dest_path,STATS,mode)
    DS.createDataJson()
    return DS
def createTFrecord(DS,STATS,mode):
    sequences=readJson(DS.action_json)
    classes=DS.class_list
    sequencesToRecord(sequences,classes,DS.ds_path,STATS,mode)
def main(args,STATS):
    start_time=time.time()
    TRAIN_DS=createDataset(args,STATS,'Train')
    EVAL_DS =createDataset(args,STATS,'Eval')
    createTFrecord(TRAIN_DS,STATS,'Train')
    createTFrecord(EVAL_DS,STATS,'Eval')

    LOG_INFO('Total Time Taken: {} s'.format(time.time()-start_time))

if __name__ == "__main__":
    main(args,STATS)
