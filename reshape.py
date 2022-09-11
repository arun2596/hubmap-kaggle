import cv2
import os 
from config import *
import pandas as pd
from model.utils import rleToMask   
from PIL import Image

df = pd.read_csv(TRAIN_CSV_FILE)
res = 768

for ind, row in df.iterrows():
    img = cv2.imread(os.path.join(STAINED_IMAGES_DIR, str(row['id']) + ".tiff" ))
    img = cv2.resize(img, (res,res))
    cv2.imwrite(os.path.join(STAINED_IMAGES_DIR_768,str(row['id']) + ".tiff" ),img)

    # mask = rleToMask(row['rle'],row['img_height'],row['img_width'])
    # mask = cv2.resize(mask, (res,res))
    # mask = 1*(mask>0)
    # cv2.imwrite(os.path.join(TRAIN_MASK_DIR_768, str(row['id']) + '_mask.tiff'),mask)



