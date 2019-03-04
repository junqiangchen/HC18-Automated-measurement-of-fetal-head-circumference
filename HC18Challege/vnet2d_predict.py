import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import DSVnet2dModule
from Vnet2d.vnet_model import Vnet2dModule
import numpy as np
import pandas as pd
import cv2


def predict_test():
    Vnet2d = Vnet2dModule(512, 768, channels=1, costname="dice coefficient", inference=True,
                          model_path="log\segmeation\model\\Vnet2d.pd")
    test_set_csv = pd.read_csv("C:\HC18\\test_set\\test_set_pixel_size.csv")
    src_test_set = "C:\HC18\\test_set\src\\"
    mask_test_set = "C:\HC18\\test_set\mask\\"
    imagedata = test_set_csv.iloc[:, 0].values
    for i in range(len(imagedata)):
        src_image = cv2.imread(src_test_set + imagedata[i], cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (768, 512))
        mask_image = Vnet2d.prediction(resize_image)
        new_mask_image = cv2.resize(mask_image, (src_image.shape[1], src_image.shape[0]))
        cv2.imwrite(mask_test_set + imagedata[i], new_mask_image)


def predict_validation():
    '''
    Preprocessing for dataset
    '''
    Vnet2d = Vnet2dModule(512, 768, channels=1, costname="dice coefficient", inference=True,
                          model_path="log\segmeation\model\\Vnet2d.pd")
    path = "C:\HC18\\training_set\process\\test\src\\"
    for i in range(990, 999, 1):
        src_image = cv2.imread(path + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (768, 512))
        mask_image = Vnet2d.prediction(resize_image)
        new_mask_image = cv2.resize(mask_image, (src_image.shape[1], src_image.shape[0]))
        cv2.imwrite(path + str(i) + "mask.bmp", new_mask_image)


#predict_validation()
predict_test()