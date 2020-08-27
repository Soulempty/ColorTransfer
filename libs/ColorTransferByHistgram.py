#coding:utf-8

import cv2
import os
import numpy as np

# Implementation of paper <<Color Transfer Based on Normalized Cumulative Hue Histograms>>。
'''
1、Base algorithm (In HSV Color Space)
2、Algorithm A+B (In HSV Color Space)
3、Algorithm A+B (In HSV Color Space)
'''
class ColorTransfer():
    
    def __init__(self,mode='base',theta=0.01):
        super(ColorTransfer,self).__init__()
        self.mode = mode
        self.theta = theta

    def CumulateHistogram(self,hist):
        cum_hist = np.zeros(hist.shape)
        c = 0
        for i in range(hist.shape[0]):
            c += hist[i]
            cum_hist[i] = c
        return cum_hist

    def ComputeHistogram(self,img,ref,r=360):
                
        img_s = img[:,:,1]/255     
        img_h = (img[:,:,0]*360/255).astype(np.int)
        ref_s = ref[:,:,1]/255
        ref_h = (ref[:,:,0]*360/255).astype(np.int)
        in_inds = img_s>0
        rn_inds = ref_s>0

        if self.mode == 'base':
            N1 = in_inds.sum()
            N2 = rn_inds.sum()
            hist = np.histogram(img_h[in_inds],r,[0,r])[0]/N1 # Normed
            ref_hist = np.histogram(ref_h[rn_inds],r,[0,r])[0]/N2 # Normed
        else:
            S1 = img_s.sum()
            S2 = ref_s.sum()
            hist = np.zeros((r))
            ref_hist = np.zeros((r))
            for i in range(r):
                 hist[i] = img_s[(img_h==i)&(in_inds)].sum()/S1
                 ref_hist[i] = ref_s[(ref_h==i)&(rn_inds)].sum()/S2
            #Remove low-frequency 
            if self.mode == 'AB':
                ref_new = np.where(ref_hist>self.theta,ref_hist,0)
                ref_hist = ref_new/ref_new.sum()

        cum_img = self.CumulateHistogram(hist)
        cum_ref = self.CumulateHistogram(ref_hist)
        return img_h,cum_img,cum_ref

    def colorTransfer(self,img,img_hist,ref_hist):
        histV = img_hist[img.ravel()]
        dif = np.abs(histV.reshape((histV.shape[0],1))-ref_hist)
        res_value = np.argmin(dif,axis=1) # Fref,norm(j)==0 is excluded.
        res = res_value.reshape(img.shape)
        return res

    def __call__(self,img,ref,range=360):
        # transform bgr to hsv
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
        ref = cv2.cvtColor(ref,cv2.COLOR_BGR2HSV_FULL)
        #compute cumulative histgram of hue
        img_h, img_hist, ref_hist = self.ComputeHistogram(img,ref,range)

        res_h = self.colorTransfer(img_h,img_hist,ref_hist)
        img[:,:,0] = (res_h*255/360).astype(np.uint8)
        image = cv2.cvtColor(img,cv2.COLOR_HSV2BGR_FULL)
        return image