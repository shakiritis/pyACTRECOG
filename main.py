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

from coreLib.utils import LOG_INFO,DataSet
#--------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Hand Action Recognition Preprocessing')
parser.add_argument("src_path", help='Source Data Folder')
parser.add_argument("dest_path", help='Destination Data Folder')
args = parser.parse_args()
#--------------------------------------------------------------------------------------------------------------------------------------------------
class STATS:
    IMAGE_DIM   =   64
    NB_CHANNELS =   3
    BATCH_SIZE  =   128
    FILE_LEN    =   512
#--------------------------------------------------------------------------------------------------------------------------------------------------
def createDataset(args,STATS,mode,EXEC):
    DS=DataSet(args.src_path,args.dest_path,STATS,mode)
    DS.create(EXEC)

def main(args,STATS):
    start_time=time.time()
    createDataset(args,STATS,'Train','tfrec')
    createDataset(args,STATS,'Test','json')
    createDataset(args,STATS,'Eval','tfrec')
    
    LOG_INFO('Total Time Taken: {} s'.format(time.time()-start_time))

if __name__ == "__main__":
    main(args,STATS)
