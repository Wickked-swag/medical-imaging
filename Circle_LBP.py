import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import skimage.feature

R = 3 #Radius
P = 8 #Pixels
impath = "pic1.png"
image = cv.imread(impath)
def use_ku():
    img2 = skimage.feature.local_binary_pattern(img1, 8, 3, method='default')
    img2 = img2.astype(np.uint8)
    hist = cv.calcHist([img2], [0], None, [256], [0, 256])
    hist = cv.normalize(hist, hist)
    plt.plot(hist, color='r')
    plt.xlim([0, 256])
    plt.show()
    plt.imshow(img2, cmap='Greys_r')
    plt.show()

def Circular_lbp(img):
    basic_array = np.zeros(image.shape,np.uint8)
    for i in range(basic_array.shape[0]-1):
        for j in range(basic_array.shape[1]-1):
            basic_array[i,j] = bin_to_decimal(cal_Circular_lbp(img,i,j))
    return basic_array
def cal_Circular_lbp(img,i,j):#比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    sum = []
    for k in range (0,P):
        x = i + R * np.cos(k * 2.0 * np.pi / P)
        y = j + R * np.sin(k * 2.0 * np.pi / P)

        x1 = int(np.floor(x)) #small
        x2 = int(np.ceil(x))  #big
        y1 = int(np.floor(y))
        y2 = int(np.ceil(y))

        #RuntimeWarning: invalid value encountered in double_scalars
        # 0/0
        # w1 = ((x2 - x) * (y2 - y)) / ((x2 - x1) * (y2 - y1))
        # w2 = ((x - x1) * (y2 - y)) / ((x2 - x1) * (y2 - y1))
        # w3 = ((x2 - x) * (y - y1)) / ((x2 - x1) * (y2 - y1))
        # w4 = ((x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))

        tx = x - x1
        ty = y - y1
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty

        f = (w1 * img[x1,y1]) + (w2 * img[x2,y1]) + (w3 * img[x1 , y2]) + (w4 * img[x2 , y2])
        if f > img[i , j]:
            sum.append(1)
        else:
            sum.append(0)
    return sum
def bin_to_decimal(bin):#二进制转十进制
    res = 0
    bit_num = 0 #左移位数
    for i in bin[::-1]:
        res += i << bit_num   # 左移n位相当于乘以2的n次方
        bit_num += 1
    return res
def show_circle_hist(a): #画lbp的直方图
    hist = cv.calcHist([a],[0],None,[256],[0,256])
    hist = cv.normalize(hist,hist)
    plt.figure(figsize = (8,4))
    plt.plot(hist, color='r')
    plt.xlim([0,256])
    plt.show()


img1 = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
#增加padding层 否则遍历的时候会出界
img1 = np.pad(img1 ,((3,2),(2,3)),'constant',constant_values = (0,0))
circle_array = Circular_lbp(img1)
show_circle_hist(circle_array)
plt.figure(figsize=(11,11))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(circle_array,cmap='Greys_r')
plt.show()
