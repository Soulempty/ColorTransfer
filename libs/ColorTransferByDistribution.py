import cv2
import numpy as np

def FastColorTransfer(img1,img2):
    img_f1 = img1.astype(np.float32)
    img_f2 = img2.astype(np.float32)

    img1_mean = img_f1[:,:,0].mean(),img_f1[:,:,1].mean(),img_f1[:,:,2].mean()
    img2_mean = img_f2[:,:,0].mean(),img_f2[:,:,1].mean(),img_f2[:,:,2].mean()

    img1_std = img_f1[:,:,0].std(),img_f1[:,:,1].std(),img_f1[:,:,2].std()
    img2_std = img_f2[:,:,0].std(),img_f2[:,:,1].std(),img_f2[:,:,2].std()

    dst  = (img1 - img1_mean)/img1_std*img2_std+img2_mean
    
    dst = np.clip(dst,0,255).astype(np.uint8)
    return dst