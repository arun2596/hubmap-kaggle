import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt

from data_loader import *
import pandas as pd

from visualization import viz_img




set_seed(123)

df_train = pd.read_csv(TRAIN_CSV_FILE)
df_train['kfold'] = 1
df_train.loc[df_train.sample(frac = 0.15).index.values,'kfold'] = 0

train_loader, valid_loader = make_loader(df_train, 2)


model = smp.Unet(
    encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=5,                      # model output channels (number of classes in your dataset)
)

model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_DIR, "model0.bin")), strict=True)

for i in valid_loader:
    img = i['image'][0,:,:,:].view(3,640,640).permute((1, 2, 0)).numpy()
    print(i['target_ind'])
    mask_shape = i['mask'].shape
    
    mask_pred = model.forward(i['image'])
    mask_pred = torch.gather(mask_pred,1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(640,640) 
    mask = torch.gather(i['mask'],1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(640,640)
    mask_pred = torch.sigmoid(mask_pred)
    #plt.imshow(mask.numpy())

    mask_pred = mask_pred.detach().numpy()
    print(mask_pred.min(),mask_pred.max())
    cv2.imshow("predicted", ((mask_pred>0.7)*255/mask_pred.max()).astype('uint8'))
    cv2.imshow("original", (mask.numpy()*255).astype('uint8'))
    # cv2.imshow("mask", mask.numpy())
    # cv2.imshow("predicted", mask_pred.numpy())

    # img2 = cv2.cvtColor(img.astype('int').astype('float32'), cv2.COLOR_RGB2BGR)
    # img2 = viz_img(img2 , res=(640,640), text=None, name="check", mask=mask.numpy())
    # cv2.imshow("check", img2.astype('uint8'))
    cv2.waitKey()
cv2.destroyAllWindows()
 

## PUSH CODE
## FUNCTION TO FIND BEST CUTOFF
## FUNCTION TO FIND DICE LOSS OF VAL SET
## AUGMENTATION - A LOT MORE
## BOUNDARY DETECTION FOR PRED CORRECTION