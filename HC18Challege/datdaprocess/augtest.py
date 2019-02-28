from  datdaprocess.Augmentation.ImageAugmentation import DataAug

aug = DataAug(rotation=20, width_shift=0.01, height_shift=0.01, rescale=1.1)
aug.DataAugmentation('data\\Train_X.csv', 'data\\Train_Y.csv', 10, path="C:\\HC18\\training_set\\process\\Aug\\")
