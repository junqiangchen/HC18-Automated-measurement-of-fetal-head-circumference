import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import DSVnet2dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('dataprocess\\data\\train_mask_aug.csv')
    csvimagedata = pd.read_csv('dataprocess\\data\\train_src_aug.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    dsVnet2d = DSVnet2dModule(768, 512, channels=1, costname="dice coefficient")
    dsVnet2d.train(imagedata, maskdata, "dsVnet2d.pd", "log\\segmeation\\", 0.001, 0.5, 10, 4)


train()
