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
import cv2

from progressbar import ProgressBar

from coreLib.utils import createLabeledImages,readJson,create_dir,LOG_INFO
#--------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Hand Action Recognition: Video Labeled Image Generation ')
parser.add_argument("json_path", help='Path for video_sec_<model>.json')
args = parser.parse_args()
FONT_FILE=os.path.join(os.getcwd(),'font.ttf')
#--------------------------------------------------------------------------------------------------------------------------------------------------
def main(args):
    _pbar=ProgressBar()
    info_sec=readJson(args.json_path)
    base_dir=os.path.dirname(args.json_path)
    base_name=str(os.path.basename(args.json_path)).replace('.json','')
    save_dir=create_dir(base_dir,base_name)
    LOG_INFO('Creating Video Sec Images')
    im_no=0
    for sec in _pbar(info_sec):
        img_path=sec['path']
        gt_id=sec['truth']
        pred_id=sec['prediction']
        createLabeledImages(img_path,gt_id,pred_id,save_dir,im_no,FONT_FILE)
        im_no+=1
    # write video
    VIDEO_NAME=os.path.join(base_dir,'{}.avi'.format(base_name))
    _mbar=ProgressBar()
    images=[os.path.join(save_dir,'seq_{}.jpg'.format(i))for i in range(0,im_no)]
    frame = cv2.imread(images[0],1)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(VIDEO_NAME, 0, 1, (width,height))
    
    LOG_INFO('Creating Video')
    for image in _mbar(images):
        video.write(cv2.imread(image,1))
    cv2.destroyAllWindows()
    video.release()
    
if __name__=='__main__':
    main(args)