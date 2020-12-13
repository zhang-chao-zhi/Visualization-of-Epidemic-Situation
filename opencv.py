#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('LOGO.png',0)    # 0以灰度模式读入图像,1读入一副彩色图像
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)           #第一个参数是窗口的名字，其次才是我们的图像
# cv2.waitKey(0)                    #检测特定键是否被按下
# k = cv2.waitKey(0)
# if k == 27:                       # wait for ESC key to exit
#  cv2.destroyAllWindows()          #轻易删除任何我们建立的窗口
# elif k == ord('q'):               # wait for 's' key to save and exit
#  cv2.imwrite('LOGO(1).png',img)   #保存图像
#  cv2.destroyAllWindows()


# #追踪蓝色物体
# cap=cv2.VideoCapture(0)
# while(1):
#   # 获取每一帧
#   ret,frame=cap.read()
#   # 转换到 HSV
#   hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#   # 设定蓝色的阈值
#   lower_blue=np.array([110,50,50])
#   upper_blue=np.array([130,255,255])
#   # 根据阈值构建掩模
#   mask=cv2.inRange(hsv,lower_blue,upper_blue)
#   # 对原图像和掩模进行位运算
#   res=cv2.bitwise_and(frame,frame,mask=mask)
#   # 显示图像
#   cv2.imshow('frame',frame)
#   cv2.imshow('mask',mask)
#   cv2.imshow('res',res)
#   k=cv2.waitKey(5)&0xFF
#   if k==ord('q'):
#      break
# # 关闭窗口
# cv2.destroyAllWindows()


# roberts边缘提取算法
#各类提取算法大同小异：sobel可控制核的大小，对于n阶，设置窗口大小为n+1，涉及平滑算子（二项展开式的系数，即n的组合数）
#差分算子，由平滑算子n-2组合数补零从后向前差分而得
#n=5阶，平滑算子14641，差分算子013310差分得，120-2-1。
import numpy as np
import cv2
import math
import sys
from scipy import signal


def roberts(I, _boundry='full', _fillvalue=0):
    # 图像的高、宽
    H1, W1 = I.shape[0:2]
    # 卷积核的尺寸
    H2, W2 = 2, 2
    # 卷积核1及锚点位置
    R1 = np.array([[1, 0], [0, -1]], np.float32)
    kr1, kc1 = 0, 0
    # 计算full卷积
    IconR1 = signal.convolve2d(I, R1, mode='full', boundary=_boundry, fillvalue=_fillvalue)
    IconR1 = IconR1[H2 - kr1 - 1:H1 + H2 - kr1 - 1, W2 - kc1 - 1:W1 + W2 - kc1 - 1]
    # 卷积核2
    R2 = np.array([[0, 1], [-1, 0]], np.float32)
    # 锚点的位置
    kr2, kc2 = 0, 1
    # 先计算full卷积
    IconR2 = signal.convolve2d(I, R2, mode='full', boundary=_boundry, fillvalue=_fillvalue)
    # 根据锚点的位置截取full卷积，从而得到same卷积
    IconR2 = IconR2[H2 - kr2 - 1:H1 + H2 - kr2 - 1, W2 - kc2 - 1:W1 + W2 - kc2 - 1]
    return (IconR1, IconR2)


if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     image = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # else:
    #     print("Usge:python roberts.py imageFile")
    # 显示原图
    image = cv2.imread('DITU.jpg', 0)
    cv2.imshow("image", image)
    # 卷积，注意边界扩充一般采取symm
    IconR1, IconR2 = roberts(image, 'symm')
    # 45度方向上的边缘强度的灰度级显示
    IconR1 = np.abs(IconR1)
    edge_45 = IconR1.astype(np.uint8)
    cv2.namedWindow('edge_45', cv2.WINDOW_NORMAL)
    cv2.imshow("edge_45", edge_45)
    # 135度方向上的边缘强度的灰度级显示
    IconR2 = np.abs(IconR2)
    edge_135 = IconR1.astype(np.uint8)
    cv2.namedWindow('edge_135', cv2.WINDOW_NORMAL)
    cv2.imshow("edge_135", edge_135)
    # 用平方和的开方来衡量最后的输出边缘
    # edge = np.sqrt(np.power(IconR1,2.0)+np.power(IconR2,2.0))
    edge = np.sqrt(np.power(IconR1, 2.0) + np.power(IconR2, 2.0))
    edge = np.round(edge)
    edge[edge > 255] = 255
    edge = edge.astype(np.uint8)
    # 显示边缘
    cv2.imshow("edge", edge)
    cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



