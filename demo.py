#coding:utf-8
import cv2
import numpy as np

from libs import FastColorTransfer,ColorTransfer

if __name__ == "__main__":
    img1 = cv2.imread('images/landscape.png')
    img2 = cv2.imread('images/dog.png')
    
    res1 = FastColorTransfer(img1,img2)
    cv2.imwrite("results/fast_res.png",res1)
    ct = ColorTransfer(mode='AB')# mode = ('base','A','AB')
    res2 = ct(img1,img2)
    cv2.imwrite("results/slow_res.png",res2)
    