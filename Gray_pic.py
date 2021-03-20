import cv2
import numpy as np
import matplotlib.pyplot  as plt

def grey_scale(image , x , x1 , y , y1):
    img_gray = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    label1 = (((img_gray <= x1) & (img_gray >=x)))
    label2 = ((img_gray > x1) & (img_gray <= 255))
    label3 = ((img_gray >= 0) & (img_gray < x))
    image[label3] = image[label3]*(y/x) + (y - (x * (y/x)))
    image[label1] = image[label1]*((y1 - y)/(x1 -x)) +(y1 -  (x1 *((y1 - y)/(x1 -x))))
    image[label2] = image[label2]*((255-y1)/(255-x1))+(y1 - (x1 * ((255-y1)/(255-x1))))
    hist = cv2.calcHist([image], [0], None, [256], [0, 255])
    plt.plot(hist)
    plt.show()
    cv2.imshow("Image" , image)
    cv2.waitKey(0)

    #k=105/215 = 0.488

def open_window(image , x , x1):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    label1 = ((img_gray >= 0) & (img_gray < x))
    label2 = ((img_gray > x1) & (img_gray <= 255))
    image[label1] = 0
    image[label2] = 0
    hist = cv2.calcHist([image], [0], None, [256], [0, 255])
    plt.plot(hist)
    plt.show()
    cv2.imshow("Image", image)
    cv2.waitKey(0)

impath = "pic1.png"
img = cv2.imread(impath)

#绘制原图以及原图灰度直方图
# hist = cv2.calcHist([img] , [0] , None , [256] , [0,255])
# plt.plot(hist)
# plt.show()


#选择骨骼灰度范围 180-210
#肌肉灰度范围40-60
#肺部灰度范围10-40

#肺部
# grey_scale(img , 50 , 60 , 30 , 150)
#骨骼
# grey_scale(img , 175 , 220 , 175 , 255)
#肌肉
# grey_scale(img , 40 ,60 ,30 , 150)

#开窗

#肺部
# open_window(img , 50 , 60)
#骨骼
# open_window(img , 175 , 220)
#肌肉
# open_window(img , 40 ,50)
