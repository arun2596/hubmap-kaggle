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

train_loader, valid_loader = make_loader(df_train, 2)


for i in train_loader:
    
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
