import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

go = np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])

def Thin(image):
    h, w = image.shape


    while(True):
        mask = np.zeros(image.shape , dtype=bool)
        for i in range (1 , h-1):
            for j in range (1 , w-1):
                if image[i][j] == 1:
                    case1 = Case1(image , i , j)
                    if(case1 >=2 and case1 <= 6):
                        case2 = Case2(image , i , j)
                        if(case2 == 1):
                            case3 = Case3(image , i , j)
                            case4 = Case4(image , i , j)
                            if(case3 and case4):
                                mask[i][j] = 1

        image[mask] = 0

        mask = np.zeros(image.shape, dtype=bool)

        for i in range (1 , h-1):
            for j in range (1 , w-1):
                if image[i][j] == 1:
                    case1 = Case1(image , i , j)
                    if(case1 >=2 and case1 <= 6):
                        case2 = Case2(image , i , j)
                        if(case2 == 1):
                            case5 = Case5(image , i , j)
                            case6 = Case6(image , i , j)
                            if(case5 and case6):
                                mask[i][j] = 1

        image[mask] = 0

        plt.imshow(image, cmap='gray_r')
        plt.show()

        if np.sum(mask) == 0:
            return image


    return image


def Case1(img_tmp , i , j):
    n = 0;
    for m in range(0,8):
        if(img_tmp[i+go[m][0]][j+go[m][1]] == 1):
            n += 1
    return n
def Case2(img_tmp , i , j):
    n = 0;
    pre = img_tmp[i + 1][j]
    for m in range(1, 8):
        if (img_tmp[i + go[m][0]][j + go[m][1]]==1 and pre == 0):
            n += 1
        pre = img_tmp[i + go[m][0]][j + go[m][1]]
    if(img_tmp[i + 1][j - 1]== 0 and img_tmp[i + 1][j] == 1):
        n += 1

    return n
def Case3(img_tmp , i , j):
    p2 = img_tmp[i][j-1]
    p4 = img_tmp[i+1][j]
    p6 = img_tmp[i][j+1]
    if p2*p4*p6 == 0:
        return True
    else:
        return False
def Case4(img_tmp , i , j):
    p8 = img_tmp[i - 1][j]
    p4 = img_tmp[i + 1][j]
    p6 = img_tmp[i][j + 1]
    if p8*p4*p6 == 0:
        return True
    else:
        return False

def Case5(img_tmp , i , j):
    p2 = img_tmp[i][j-1]
    p4 = img_tmp[i+1][j]
    p8 = img_tmp[i - 1][j]
    if p2*p4*p8 == 0:
        return True
    else:
        return False
def Case6(img_tmp , i , j):
    p8 = img_tmp[i - 1][j]
    p2 = img_tmp[i][j-1]
    p6 = img_tmp[i][j + 1]
    if p8*p2*p6 == 0:
        return True
    else:
        return False
# 读取灰度图片，并显示
img = cv2.imread('图片1.png' , 0)  # 直接读为灰度图像
h, w = img.shape
img = img[ : , 1 : w-1]
img1 = np.pad(img ,((1,1),(1,1)),'constant',constant_values = (0,0))
ret , dst = cv2.threshold(img1 , 127 , 255 , cv2.THRESH_BINARY)
dst = dst / 255
thin = Thin(dst)
plt.imshow(thin , cmap='gray_r')
plt.savefig('图片1的结果.png')
plt.show()
