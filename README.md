# pyACT
    Version: 0.0.1    
    Author : Md. Nazmuddoha Ansary
                  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements
* tensorflow==1.13.1        
* Python == 3.6.8
> Create a Virtualenv and *pip3 install -r requirements.txt*

# DataSet 

Dataset is taken from [Hand Action Detection from Depth Sequences](https://web.bii.a-star.edu.sg/archive/machine_learning/Projects/behaviorAnalysis/handAction_ECHT/index.htm)
> Members:Chi Xu,Lakshmi Narasimhan Govindarajan,Li Cheng 

* Get a general idea about the [dataset info](/info/data.md) 
* The **Train1** and **Train2** are combined in one single folder named **Train**
* The **Test** and **TestDetection** from the original source are kept and all three folders are to be in the same directory which should have the following folder tree:

            ├── Test
            │   ├── alvin
            │   │   ├── asl-bathroom
            │   │   ├── asl-blue
            ....................
            │   │   ├── ui-keyTap
            │   │   ├── ui-screenTap
            │   │   └── ui-swipe
            │   ├── etienne
            ................
            │   └── xiaowei
            ...............
            ├── TestDetection
            │   ├── alvin_test
            ..................
            │   ├── xiaowei_test
            │   └── zhaohe
            └── Train
                ├── aiyang
                │   ├── asl-bathroom
                .....................
                .....................
                │   ├── ui-doubleclick
                │   ├── ui-keyTap
                │   ├── ui-screenTap
                │   └── ui-swipe
                └── yongzhong
                    ├── asl-bathroom
                    .................
                    ├── ui-screenTap
                    └── ui-swipe


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


            ├── Test
            │   ├── action.json
            │   └── info.json
            ├── TFRECORD
            │   ├── Test
            │   │   ├── Test_0.tfrecord
            ...........................
            ...........................
            ...........................
            │   │   └── Test_X.tfrecord
            │   └── Train
            │       ├── Train_0.tfrecord
            ............................
            ............................
            ............................
            │       └── Train_X.tfrecord
            └── Train
                ├── action.json
                └── info.json


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

# TODO

* Add CNN-LSTM 