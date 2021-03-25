nvidia-docker run -it --ipc=host -v /root/ssy:/root/ssy --name ssyDIEN10  nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04  
#docker pull nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
#nvidia-docker run -it --ipc=host -v /root/ssy:/root/ssy --name ssyDIEN   nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
nvidia-docker start ssyDIEN10
nvidia-docker exec -it ssyDIEN10 /bin/bash

source /root/ssy/ai/training/env.sh
apt update
apt install python3 -y
apt install python3-pip -y
# first install this version,
# or else if you install tensorflow-gpu first, it will use newer numpy requiring python3.7
pip3 install numpy==1.14.5
pip3 install h5py==2.7.0
pip3 install futures==2.2.0
pip3 install tensorflow-gpu==1.14.0




# Deep Interest Evolution Network for Click-Through Rate Prediction
https://arxiv.org/abs/1809.03672
## prepare data
### method 1
You can get the data from amazon website and process it using the script
```
sh prepare_data.sh
```
### method 2 (recommended)
Because getting and processing the data is time consuming，so we had processed it and upload it for you. You can unzip it to use directly.
```
tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```
When you see the files below, you can do the next work. 
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info
## train model
```
python train.py train [model name] 
SSY's version
python3 ./script/train.py train DIEN
```
The model blelow had been supported: 
- DNN 
- PNN 
- Wide (Wide&Deep NN) 
- DIN  (https://arxiv.org/abs/1705.06978) 
- DIEN (https://arxiv.org/pdf/1809.03672.pdf) 

Note: we use tensorflow 1.4.
