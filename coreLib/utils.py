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
#--------------------------------------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    def __init__(self,src_path,dest_path,STATS,mode):
        self.src_path  = src_path
        self.dest_path = dest_path
        self.mode = mode
        self.ds_path=create_dir(self.dest_path,'DataSet')
        self.ds_path=create_dir(self.ds_path,self.mode)
        self.info_json=os.path.join(self.ds_path,'info.json')
        self.action_json=os.path.join(self.ds_path,'action.json')
        
        self.image_dim  =   STATS.IMAGE_DIM 
        self.nb_channels=   STATS.NB_CHANNELS
        self.batch_size =   STATS.BATCH_SIZE

    
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
                
                seq_dict={'path' :img_dir,
                          'start':id_start,
                          'stop' :id_stop,
                          'class':final_class}
                if final_class=='no-action':
                    not_dict.append(seq_dict)
                else:
                    action_dict.append(seq_dict)

            if self.mode!='Test':
                not_len = len(action_dict) // len(self.class_list)
                random.shuffle(not_dict)
                not_dict=not_dict[:not_len]
            
            data_dict=action_dict+not_dict
            
            if self.mode!='Test':
                random.shuffle(data_dict)
                data_len =  (len(data_dict) // self.batch_size) * self.batch_size
                data_dict=data_dict[:data_len] 
            
            dump_data(self.action_json,data_dict)
            self.nb_seqs=len(data_dict)

    def createDataJson(self):
        self.__getMinSeqLen()
        self.__getClasses()
        self.__setSeqLabels()
        data={'nb_classes':len(self.class_list),
               'seq_len':self.min_seq_len,
               'image_dim':self.image_dim,
               'nb_channels':self.nb_channels,
               'nb_seqs': self.nb_seqs}
               #'class_ids':self.class_list}]
        dump_data(self.info_json,data)
#--------------------------------------------------------------------------------------------------------------------------------------------------
def read_image(image_path,image_dim):
    img=tf.keras.preprocessing.image.load_img(image_path,target_size=(image_dim,image_dim),color_mode='grayscale')
    x=tf.keras.preprocessing.image.img_to_array(img,dtype=np.uint8)
    x=np.expand_dims(x,axis=-1)
    return x
def preprocess_sequence(sequnce_data,classes,STATS,img_iden='color_',img_ext='.png'):
    img_dir=sequnce_data['path']
    id_start=int(sequnce_data['start'])
    id_stop=int(sequnce_data['stop'])
    class_id=sequnce_data['class']
    image_paths=[os.path.join(img_dir,'{}{}{}'.format(img_iden,i,img_ext)) for i in range(id_start,id_stop+1)]
    features=[]
    for image_path in image_paths:
        features.append(read_image(image_path,image_dim=STATS.IMAGE_DIM))
    X=np.array(features)
    Y=classes.index(class_id)
    return X,Y
def sequencesToRecord(sequences,classes,ds_dir,STATS,mode):
    LOG_INFO('Creating TFrecords:{}'.format(mode))
    FS=STATS.FILE_LEN
    for i in range(0,len(sequences),FS):
        sequence_list= sequences[i:i+FS]        
        rec_num = i // FS
        LOG_INFO('REC NUM:{}'.format(rec_num))
        saveTFrecord(ds_dir,rec_num,sequence_list,mode,classes,STATS)
#--------------------------------------------------------------------------------------------------------------------------------------------------
def saveTFrecord(ds_dir,rec_num,sequence_list,mode,classes,STATS):
    _pbar=ProgressBar()
    # Create Saving Directory based on mode
    base_ds_dir=os.path.dirname(ds_dir)
    save_dir=create_dir(base_ds_dir,'TFRECORD')
    save_dir=create_dir(save_dir,mode)
    tfrecord_name='{}_{}.tfrecord'.format(mode,rec_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name) 
    # writting to tfrecords
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for seqence in _pbar(sequence_list):
            X,Y=preprocess_sequence(seqence,classes,STATS)
            # feature desc
            data ={ 'feats':tf.train.Feature(int64_list=tf.train.Int64List(value=X.flatten())),
                    'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[Y]))
            }
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)   
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
        data  ={ 'feats':tf.io.FixedLenFeature((FLAGS.MIN_SEQ_LEN,FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS),tf.int64),
                 'label':tf.io.FixedLenFeature((),tf.int64)
        }    
        parsed_example=tf.io.parse_single_example(example,data)
        feats=tf.cast(parsed_example['feats'],tf.float32)/255.0
        feats=tf.reshape(feats,(FLAGS.MIN_SEQ_LEN,FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
        
        idx = tf.cast(parsed_example['label'], tf.int64)
        label=tf.one_hot(idx,FLAGS.NB_CLASSES,dtype=tf.int64)
        return feats,label

    file_paths=glob(os.path.join(FLAGS.TFRECORDS_DIR,'{}*.tfrecord'.format(FLAGS.MODE)))
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(FLAGS.SHUFFLE_BUFFER,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.BATCH_SIZE,drop_remainder=True)
    return dataset
