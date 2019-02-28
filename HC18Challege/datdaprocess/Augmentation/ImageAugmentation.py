from datdaprocess.Augmentation.images_masks_transform import ImageDataGenerator
import cv2
import pandas as pd
import os

'''
Feature Standardization
standardize pixel values across the entire dataset
ZCA Whitening
A whitening transform of an image is a linear algebra operation that reduces the redundancy in the matrix of pixel images.
Less redundancy in the image is intended to better highlight the structures and features in the image to the learning algorithm.
Typically, image whitening is performed using the Principal Component Analysis (PCA) technique.
More recently, an alternative called ZCA (learn more in Appendix A of this tech report) shows better results and results in
transformed images that keeps all of the original dimensions and unlike PCA, resulting transformed images still look like their originals.
Random Rotations
sample data may have varying and different rotations in the scene.
Random Shifts
images may not be centered in the frame. They may be off-center in a variety of different ways.
RESCALE
对图像按照指定的尺度因子, 进行放大或缩小, 设置值在0- 1之间，通常为1 / 255;
Random Flips
improve performance on large and complex problems is to create random flips of images in your training data.
fill_mode: 填充像素, 出现在旋转或平移之后．
'''


class DataAug(object):
    '''
    transform Image and Mask together
    '''

    def __init__(self, rotation=5, width_shift=0.05,
                 height_shift=0.05, rescale=1.2, horizontal_flip=True, vertical_flip=False):
        # define data preparation
        self.__datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,
                                            zca_whitening=False,
                                            rotation_range=rotation, width_shift_range=width_shift,
                                            height_shift_range=height_shift, rescale=rescale,
                                            horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                            fill_mode='nearest')

    def __ImageMaskTranform(self, images, labels, index, number, path):
        # reshape to be [samples][pixels][width][height][channels]
        images = cv2.imread(images, 1)
        labels = cv2.imread(labels, 1)
        srcimage = images.reshape([1, images.shape[0], images.shape[1], images.shape[2]])
        srclabel = labels.reshape([1, labels.shape[0], labels.shape[1], labels.shape[2]])

        i = 0
        for batch1, batch2 in self.__datagen.flow(srcimage, srclabel):
            i += 1
            batch1 = batch1[0, :, :, :]
            src_path = path + 'src\\'
            if not os.path.exists(src_path):
                os.makedirs(src_path)
            cv2.imwrite(src_path + str(index) + '_' + str(i) + '.bmp', batch1)

            batch2 = batch2[0, :, :, :]
            mask_path = path + 'mask\\'
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
            cv2.imwrite(mask_path + str(index) + '_' + str(i) + '.bmp', batch2)

            if i > number - 1:
                break

    def DataAugmentation(self, filepathX, filepathY, number=100, path=None):
        csvXdata = pd.read_csv(filepathX)
        dataX = csvXdata.iloc[:, :].values

        csvYdata = pd.read_csv(filepathY)
        dataY = csvYdata.iloc[:, :].values
        for index in range(dataX.shape[0]):
            # For images
            images = dataX[index][0]
            # For labels
            labels = dataY[index][0]
            self.__ImageMaskTranform(images, labels, index, number, path)
