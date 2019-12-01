# pyACT
    Version: 0.0.6    
    Author : Md. Nazmuddoha Ansary
                  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements
* tensorflow==1.13.1
* numpy==1.16.4        
* Python == 3.6.8
> Create a Virtualenv and *pip3 install -r requirements.txt*

# DataSet 

Dataset is taken from [Hand Action Detection from Depth Sequences](https://web.bii.a-star.edu.sg/archive/machine_learning/Projects/behaviorAnalysis/handAction_ECHT/index.htm)
> Members:Chi Xu,Lakshmi Narasimhan Govindarajan,Li Cheng 

* Get a general idea about the [dataset info](/info/data.md) 
* The **Train1** and **Train2** are combined in one single folder named **Train**
* A random user is chosen for **Eval** Data Within **Train** data
* The **Test** the original source are kept

> Folder tree for the **src_path** for **main.py** in implementation case is as follows:


            ├── Eval
            │   └── LN
            ├── Test
            │   ├── alvin
            │   ├── etienne
            │   └── xiaowei
            └── Train
                ├── aiyang
                ├── ashwin
                ├── biru
                ├── chi
                ├── chris
                ├── gulin
                ├── justin
                ├── lakshmi
                ├── malay
                ├── marc
                ├── michael
                ├── xuchi
                ├── yizhou
                └── yongzhong

            

* run **main.py**

            usage: main.py [-h] src_path dest_path

            Hand Action Recognition Preprocessing

            positional arguments:
            src_path    Source Data Folder
            dest_path   Destination Data Folder

            optional arguments:
            -h, --help  show this help message and exit

##### NOTE
The complete preprocessing may take huge time and also cause to crash the system due to high memory useage. A way around is built for **Ubuntu** users is to run **sudo ./clear_mem.sh** in parallel with **main.py**

* After execution, the provided **dest_path** should have a **DataSet** folder with the following folder tree:


            ├── mode:Eval_numOfSeqences:3456_minSeqLen:6.json
            ├── mode:Test_numOfSeqences:21785_minSeqLen:6.json
            ├── mode:Train_numOfSeqences:49920_minSeqLen:6.json
            └── TFRECORD
                ├── Eval
                │   ├── Eval_0.tfrecord
                │   ├── Eval_1.tfrecord
                │   ├── Eval_2.tfrecord
                │   ├── Eval_3.tfrecord
                │   └── Eval_4.tfrecord
                └── Train
                    ├── Train_0.tfrecord
                    ├── Train_10.tfrecord
                    ├── Train_11.tfrecord
                    ├── Train_12.tfrecord
                    ├── Train_13.tfrecord
                    ├── Train_14.tfrecord
                    ├── Train_15.tfrecord
                    ├── Train_16.tfrecord
                    ├── Train_17.tfrecord
                    ├── Train_18.tfrecord
                    ├── Train_19.tfrecord
                    ├── Train_1.tfrecord
                    ├── Train_20.tfrecord
                    ├── Train_21.tfrecord
                    ├── Train_22.tfrecord
                    ├── Train_23.tfrecord
                    ├── Train_24.tfrecord
                    ├── Train_25.tfrecord
                    ├── Train_26.tfrecord
                    ├── Train_27.tfrecord
                    ├── Train_28.tfrecord
                    ├── Train_29.tfrecord
                    ├── Train_2.tfrecord
                    ├── Train_30.tfrecord
                    ├── Train_31.tfrecord
                    ├── Train_32.tfrecord
                    ├── Train_33.tfrecord
                    ├── Train_34.tfrecord
                    ├── Train_35.tfrecord
                    ├── Train_36.tfrecord
                    ├── Train_37.tfrecord
                    ├── Train_38.tfrecord
                    ├── Train_3.tfrecord
                    ├── Train_4.tfrecord
                    ├── Train_5.tfrecord
                    ├── Train_6.tfrecord
                    ├── Train_7.tfrecord
                    ├── Train_8.tfrecord
                    └── Train_9.tfrecord

* 3 directories, 47 files



**ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  


# TPU(Tensor Processing Unit)
![](/info/src_img/tpu.ico?raw=true)*TPU’s have been recently added to the Google Colab portfolio making it even more attractive for quick-and-dirty machine learning projects when your own local processing units are just not fast enough. While the **Tesla K80** available in Google Colab delivers respectable **1.87 TFlops** and has **12GB RAM**, the **TPUv2** available from within Google Colab comes with a whopping **180 TFlops**, give or take. It also comes with **64 GB** High Bandwidth Memory **(HBM)**.*
[Visit This For More Info](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da)  

#  GCS (Google Cloud Storage)	
![](/info/src_img/bucket.ico?raw=true) Training with tfrecord is not implemented for local implementation in colab.	
For using colab, a **bucket** must be created in **GCS** and connected for:
* tfrecords
* checkpoints (custom training Loop)

# MODELS
## CONVNET3D:
The model is based on the paper [Learning Spatiotemporal Features with 3D Convolutional Networks](https://ieeexplore.ieee.org/document/7410867)  
An adapted model structre is as follow:

![](/info/convNet3D.png?raw=true)

