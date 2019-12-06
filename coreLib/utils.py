# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import os
import numpy as np 
import matplotlib.pyplot as plt

from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import json
import random
from glob import glob
from progressbar import ProgressBar

import tensorflow as tf

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

#---------------------------------------------------------------------------
def readJson(file_name):
    return json.load(open(file_name))

def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',data=data)
    hf.close()

def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

def LOG_INFO(log_text,p_color='green',rep=True):
    if rep:
        print(colored('#    LOG:','blue')+colored(log_text,p_color))
    else:
        print(colored('#    LOG:','blue')+colored(log_text,p_color),end='\r')

def create_dir(base_dir,ext_name):
    new_dir=os.path.join(base_dir,ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def dump_data(file_path,data):
    with open(file_path,'w') as fd:
        json.dump(data,fd,indent=2,ensure_ascii=False)

def read_image(image_path,image_dim,rot_angle=None,flip_id=None):
    img=Image.open(image_path)
    img=img.resize((image_dim,image_dim))
    if rot_angle:
        img=img.rotate(rot_angle)
    if flip_id:
        x=getFlipDataById(img,flip_id)
    else:
        x=getFlipDataById(img,0)
    x=(x/255.0).astype(np.float32)
    return x

def preprocess_sequence(sequnce,class_list,image_dim,img_iden='color_',img_ext='.png'):
    img_dir=sequnce['path']
    id_start=int(sequnce['start'])
    id_stop=int(sequnce['stop'])
    class_id=sequnce['class']
    rot_angle=sequnce['rotation']
    flip_id=int(sequnce['flip'])
    image_paths=[os.path.join(img_dir,'{}{}{}'.format(img_iden,i,img_ext)) for i in range(id_start,id_stop+1)]
    features=[]
    for image_path in image_paths:
        features.append(read_image(image_path,image_dim,rot_angle,flip_id))
    X=np.array(features)
    Y=class_list.index(class_id)
    return X,Y

def getFlipDataById(img,fid):
    if fid==0:# ORIGINAL
        x=np.array(img)
    elif fid==1:# Left Right Flip
        x=np.array(img.transpose(Image.FLIP_LEFT_RIGHT))
    elif fid==2:# Up Down Flip
        x=np.array(img.transpose(Image.FLIP_TOP_BOTTOM))
    else: # Mirror Flip
        x=img.transpose(Image.FLIP_TOP_BOTTOM)
        x=np.array(x.transpose(Image.FLIP_LEFT_RIGHT))
    return x
#--------------------------------------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    def __init__(self,src_path,dest_path,STATS,mode,rot_start=0,rot_step=5,rot_stop=40):
        self.src_path  = src_path
        self.dest_path = dest_path
        self.mode = mode
        self.ds_path=create_dir(self.dest_path,'DataSet')
        
        
        self.image_dim  =   STATS.IMAGE_DIM 
        self.nb_channels=   STATS.NB_CHANNELS
        self.batch_size =   STATS.BATCH_SIZE
        self.file_size  =   STATS.FILE_LEN

        self.rot_start=rot_start
        self.rot_stop=rot_stop
        self.rot_step=rot_step

    
    def __getMinSeqLen(self):
        min_seq_len=1000
        for filename in Path(self.src_path).rglob('*.xml'):
            root = ET.parse(filename).getroot()
            for item in root.iter("len"):
                min_seq_len=min(min_seq_len,int(item.text))
        self.min_seq_len=min_seq_len
        
    def __getClasses(self):
        self.class_list=[]
        user_handle=os.listdir(os.path.join(self.src_path,self.mode))[0]
        item_list =os.listdir(os.path.join(self.src_path,self.mode,user_handle))
        for item in item_list:
            if '.xml' not in item:
                self.class_list.append(item)
        self.class_list.append('no-action')
        self.nb_classes=len(self.class_list)

    def __setSeqLabels(self):
        action_dict=[]
        not_dict=[]
        data_dict=[]
        _pbar=ProgressBar()
        file_list=[]
        LOG_INFO('Creating Sequence Data Json for {}'.format(self.mode))
        self.src_path=os.path.join(self.src_path,self.mode)
        for filename in Path(self.src_path).rglob('*.xml'):
            file_list.append(filename)
            
        for filename in _pbar(file_list):
            img_dir=str(filename).replace('.xml','')
            img_len=len(glob(os.path.join(img_dir,'color_*.png')))
            class_name=os.path.basename(img_dir)
            root = ET.parse(filename).getroot()
            _lengths=[]
            _start_ids=[]
            _img_no_ids=[]
            for item in root.iter("_"):
                for child in item:
                    if child.tag=='len':
                        _lengths.append(int(child.text))
                    if child.tag=='pos':
                        _start_ids.append(int(child.text))
            for i in range(len(_start_ids)):
                id_start=_start_ids[i]
                id_stop =id_start+_lengths[i]
                _img_no_ids+=[id_val for id_val in range(id_start,id_stop)]
            for i in range(self.min_seq_len-1,img_len):
                id_stop=i
                id_start=i+1-self.min_seq_len
                seq_ids=[id_val for id_val in range(id_start,id_stop+1)]
                match_len=len(list(set(seq_ids) & set(_img_no_ids)))
                if match_len >= self.min_seq_len//2:
                    final_class=class_name
                else:
                    final_class='no-action'
                
                if self.mode!='Test': 
                    for rot_angle in range(self.rot_start,self.rot_stop,self.rot_step):
                        for flip_id in range(4):
                            seq_dict={'path' :img_dir,
                                    'start':id_start,
                                    'stop' :id_stop,
                                    'class':final_class,
                                    'rotation':rot_angle,
                                    'flip':flip_id}
                            if final_class=='no-action':
                                not_dict.append(seq_dict)
                            else:
                                action_dict.append(seq_dict)
                else:
                    seq_dict={'path' :img_dir,
                                    'start':id_start,
                                    'stop' :id_stop,
                                    'class':final_class,
                                    'rotation':0,
                                    'flip':0}
                    data_dict.append(seq_dict)
            
            if self.mode!='Test':
                not_len = len(action_dict) // len(self.class_list)
                not_dict=not_dict[:not_len]
                data_dict=action_dict+not_dict
                data_len =  (len(data_dict) // self.batch_size) * self.batch_size
                data_dict=data_dict[:data_len]
                random.shuffle(data_dict)
        
        self.nb_seqs=len(data_dict)
        self.json_path=os.path.join(self.ds_path,'mode:{}_numOfSeqences:{}_minSeqLen:{}.json'.format(self.mode,self.nb_seqs,self.min_seq_len)) 
        dump_data(self.json_path,data_dict)
            

    def __createDataJson(self):
        self.__getMinSeqLen()
        self.__getClasses()
        self.__setSeqLabels()

    
    def __saveTFrecord(self,rec_num,sequence_list):
        _pbar=ProgressBar()
        # Create Saving Directory based on mode
        save_dir=create_dir(self.ds_path,'TFRECORD')
        save_dir=create_dir(save_dir,self.mode)
        tfrecord_name='{}_{}.tfrecord'.format(self.mode,rec_num)
        tfrecord_path=os.path.join(save_dir,tfrecord_name) 
        # writting to tfrecords
        with tf.io.TFRecordWriter(tfrecord_path) as writer:    
            for seqence in _pbar(sequence_list):
                X,Y=preprocess_sequence(seqence,self.class_list,self.image_dim)
                # feature desc
                data ={ 'feats':tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
                        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[Y]))
                }
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)   

    def __sequencesToRecord(self):
        LOG_INFO('Creating TFrecords:{}'.format(self.mode))
        FS=self.file_size
        sequences=readJson(self.json_path)
        for i in range(0,len(sequences),FS):
            sequence_list= sequences[i:i+FS]        
            rec_num = i // FS
            LOG_INFO('REC NUM:{}'.format(rec_num))
            self.__saveTFrecord(rec_num,sequence_list)

    def create(self,EXEC):
        if EXEC=='json':
            self.__createDataJson()
        else:
            self.__createDataJson()
            self.__sequencesToRecord()

#--------------------------------------------------------------------------------------------------------------------------------------------------
def data_input_fn(FLAGS): 
    '''
    This Function generates data from provided FLAGS
    FLAGS must include:
        TFRECORDS_DIR  = Path to tfrecords
        IMAGE_DIM       = Dimension of Image
        NB_CHANNELS     = Depth of Image
        BATCH_SIZE      = batch size for traning
        SHUFFLE_BUFFER  = Buffer Size > Batch Size
        MODE            = 'Train/Eval'
        NB_CLASSES      = Number of classes
        MIN_SEQ_LEN     = Minimul Seqlen for the data
    '''
    
    def _parser(example):
        data  ={ 'feats':tf.io.FixedLenFeature((FLAGS.MIN_SEQ_LEN,FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS),tf.float32),
                 'label':tf.io.FixedLenFeature((),tf.int64)
        }    
        parsed_example=tf.io.parse_single_example(example,data)
        feats=tf.cast(parsed_example['feats'],tf.float32)
        feats=tf.reshape(feats,(FLAGS.MIN_SEQ_LEN,FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
        
        idx = tf.cast(parsed_example['label'], tf.int64)
        label=tf.one_hot(idx,FLAGS.NB_CLASSES,dtype=tf.int64)
        return feats,label

    file_paths=glob(os.path.join(FLAGS.TFRECORDS_DIR,FLAGS.MODE,'{}*.tfrecord'.format(FLAGS.MODE)))
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(FLAGS.SHUFFLE_BUFFER,reshuffle_each_iteration=True)
    dataset = dataset.batch(FLAGS.BATCH_SIZE,drop_remainder=True)
    return dataset
#--------------------------------------------------------------------------------------------------------------------------------------------------
def createLabeledImages(img_path,gt_id,pred_id,save_dir,im_no,font_file):
    # pad image
    img=tf.keras.preprocessing.image.load_img(img_path)
    arr=tf.keras.preprocessing.image.img_to_array(img,dtype=np.uint8)
    pad=abs(arr.shape[0]-arr.shape[1])//2
    if arr.shape[0] > arr.shape[1]:
        pad_axis=1
        dim2=pad
        dim1=arr.shape[0]
        TFIT=arr.shape[1]
    else:
        pad_axis=0
        dim1=pad
        dim2=arr.shape[1]
        TFIT=arr.shape[0]
    arr_pad=np.zeros((dim1,dim2,arr.shape[-1]),dtype=np.uint8)
    new_arr=np.concatenate((arr,arr_pad),axis=pad_axis)
    # draw text
    img=Image.fromarray(new_arr)
    draw = ImageDraw.Draw(img)
    FONT_SIZE=pad//2
    font = ImageFont.truetype(font_file, FONT_SIZE)
    TEXT_PRED='PREDICTION: {}'.format(gt_id)
    TEXT_GT  ='GROUND_TRUTH: {}'.format(pred_id)
    draw.text((0, TFIT),TEXT_GT,(255,255,0),font=font)
    if pred_id==gt_id:
        col=(0,255,0)
    else:
        col=(255,0,0)
    draw.text((0, TFIT+FONT_SIZE),TEXT_PRED,col,font=font)
    # save image
    img.save(os.path.join(save_dir,'seq_{}.jpg'.format(im_no)))
