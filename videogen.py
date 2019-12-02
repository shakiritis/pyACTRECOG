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
parser.add_argument("max_len", help='max number of images used to create one video file')
args = parser.parse_args()
FONT_FILE=os.path.join(os.getcwd(),'font.ttf')
#--------------------------------------------------------------------------------------------------------------------------------------------------
def main(args):
    _pbar=ProgressBar()
    info_sec=readJson(args.json_path)
    base_dir=os.path.dirname(args.json_path)
    base_dir=create_dir(base_dir,'Results')
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
    _mbar=ProgressBar()
    images=[os.path.join(save_dir,'seq_{}.jpg'.format(i))for i in range(0,im_no)]
    frame = cv2.imread(images[0],1)
    height, width, layers = frame.shape
    
    LOG_INFO('Creating Video')
    image_len=args.max_len
    for i in _mbar(range(0,len(images),image_len)):
        path_list=images[i:i+image_len]
        vid_num= i % image_len
        VIDEO_NAME=os.path.join(base_dir,'{}_{}.avi'.format(base_name,vid_num))
        video = cv2.VideoWriter(VIDEO_NAME, 0, 1, (width,height))
        for image in path_list:
            video.write(cv2.imread(image,1))
        cv2.destroyAllWindows()
        video.release()
    
if __name__=='__main__':
    main(args)