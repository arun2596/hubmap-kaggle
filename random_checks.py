import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt

from data_loader import *
import pandas as pd

from infer_utils import getDiceLoss

from visualization import viz_img

df_train = pd.read_csv(TRAIN_CSV_FILE)
df_train['kfold'] = 1
df_train.loc[df_train.sample(frac = 0.15).index.values,'kfold'] = 0

import json

# print(df_train.columns)
# df_train['ind'] = df_train.apply(lambda x: CLASS_TO_ID[x['organ']]-1, axis=1)

# print(df_train['ind'].value_counts())
# print(df_train['data_source'].unique())
# print(df_train[['ind','kfold','organ']].groupby(['ind','kfold']).count())



train_loader, valid_loader = make_loader(df_train, 2)

# for i in train_loader:
#     print(i['image'].shape)
#     break


for i in train_loader:
    # if i['target_ind'].numpy()[0]!=2:
    #     continue
    img = i['image'][0,:,:,:].view(3,640,640).permute((1, 2, 0)).numpy()
    mask_shape = i['mask'].shape
    mask = torch.gather(i['mask'],1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(640,640)
    
    # mask_pred = torch.gather(mask_pred,1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(640,640) 
    # mask = torch.gather(i['mask'],1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(640,640)

    #plt.imshow(mask.numpy())
    

    # mask_pred = mask_pred.detach().numpy()

    # cv2.imshow("predicted", ((mask_pred>0.1)*255/mask_pred.max()).astype('uint8'))
    # cv2.imshow("original", (mask.numpy()*255).astype('uint8'))
    cv2.imshow("mask", mask.numpy().astype('uint8')*255)
    cv2.imshow("img",img.astype('uint8'))

    img2 = viz_img(img.astype('uint8') , res=(640,640), text=None, name="check", mask=mask.numpy().astype('uint8'))
    cv2.imshow("mask_img", img2)
    cv2.waitKey()
    cv2.destroyAllWindows()


# checking lung images

# df_train = df_train[df_train['organ']=='lung']


# for x,i in df_train.iterrows():

#     mask = rleToMask(i['rle'],i['img_height'], i['img_width'])
#     mask = mask.reshape(i['img_height'], i['img_width'],1)
    
#     with open('data/raw/train_annotations/'+str(i['id'])+'.json') as f:
#         pol = json.load(f)
    
#     cv2.imshow('mask',cv2.resize(mask.astype('uint8')*255, (1500,1500)))
#     img = cv2.imread('data/raw/train_images/'+str(i['id'])+'.tiff')
#     cv2.putText(img, str(len(pol)) + ' ' + i['sex'] + " " + str(i['age']) + " " + str(i['id']), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
#     cv2.imshow('image',cv2.resize(img,(1500,1500)))
#     cv2.waitKey()
# cv2.destroyAllWindows()