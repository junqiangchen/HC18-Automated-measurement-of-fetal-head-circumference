# HC18-Automated-measurement-of-fetal-head-circumference
> This is an example of the ultrasound imaging is used to measure the fetal head circumference (HC).
![](bannerV3_V5OH10E.png)

## How to Use

**1、Preprocess**
* download ultrasound image data:https://zenodo.org/record/1327317
* convert annotation image to mask image,you can download here:https://pan.baidu.com/s/1NmmztKR0Omu1fKue7263Sw password：6ax7 
* data augmentation:run the augtest.py

**2、HC Segmentation**
* the VNet2d model

![](VNet2d.png) 

* train and predict in the script of vnet2d_train.py and vnet2d_predict.py

**3、trained model can download on here:https://pan.baidu.com/s/1ved3w6Tn4nwnBUDzxW8arw password：r0li **

## Result
![](37.bmp)
![](38.bmp)
![](32.bmp)

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* Contact: junqiangChen
* WeChat Number: 1207173174
* WeChat Public number: 最新医学影像技术
