import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import *
from  model.utils import rleToMask, maskToRle

def addMaskToImage(img , mask):
    """add mask to image"""
    img[:,:,-1] = np.where(mask, img[:,:,-1]*0.1 + mask*0.9*255, img[:,:,-1]) 
    img[:,:, 1] = np.where(mask, img[:,:, 1]*0.5, img[:,:,1]) 
    img[:,:, 0] = np.where(mask, img[:,:, 0]*0.5, img[:,:, 0]) 
    return img

def viz_img(img , res=(1280,1280), text=None, name="default", mask=None):
    """ adds text, resize img and add mask"""
    if mask is not None:
        img = addMaskToImage(img,mask)
    
    img = cv2.resize(img,res)
    if text is not None:
        cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
    return img

# df_train = pd.read_csv(TRAIN_CSV_FILE)
# print(df_train['organ'].unique())
# # df_train = df_train[df_train['organ']=='largeintestine'].reset_index(drop=True)

# offset = 0


# # for ind in range(df_train.shape[0]):
# #     if ind < offset:
# #         continue
# #     file = str(df_train['id'][ind]) + ".tiff"
# #     img_shape = (int(df_train['img_height'][ind]), int(df_train['img_width'][ind]))
# #     print(file)
# #     mask = rleToMask(df_train['rle'][ind], *img_shape)
# #     rle = maskToRle(mask)
# #     print(rle == df_train['rle'][ind])
# #     img = cv2.imread(os.path.join(TRAIN_IMAGES_DIR,file),cv2.IMREAD_COLOR)
# #     res = 640

#     # cv2.imshow('original', cv2.resize(img, (res,res)))

#     # img[:,:,-1] = np.where(mask, img[:,:,-1]*0.1 + mask*0.9*255, img[:,:,-1]) 
#     # img[:,:, 1] = np.where(mask, img[:,:, 1]*0.5, img[:,:,1]) 
#     # img[:,:, 0] = np.where(mask, img[:,:, 0]*0.5, img[:,:, 0]) 
#     # img = cv2.resize(img, (res,res))
#     # cv2.putText(img, df_train['organ'][ind], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
#     # cv2.imshow('masked', img)
#     # cv2.waitKey()
# cv2.destroyAllWindows()