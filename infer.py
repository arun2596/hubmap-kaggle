import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt

from data_loader import *
import pandas as pd

from infer_utils import getDiceLoss

from visualization import viz_img




set_seed(123)

df_train = pd.read_csv(TRAIN_CSV_FILE)
df_train['kfold'] = 1
df_train.loc[df_train.sample(frac = 0.15).index.values,'kfold'] = 0

train_loader, valid_loader = make_loader(df_train, 2, input_shape=640)


model = smp.Unet(
    encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=5,                      # model output channels (number of classes in your dataset)
)

thresholds = [0.05,0.1,0.2,0.3, 0.4, 0.45, 0.5, 0.55,0.6,0.7,0.8,0.9]

model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_DIR, "model0.bin")), strict=True)
model = model.cuda()
model.eval()
all_losses = None
target_ls = []
for i in valid_loader:
    #img = i['image'][0,:,:,:].view(3,640,640).permute((1, 2, 0)).numpy()
  
    mask_shape = i['mask'].shape
    mask = i['mask'].cuda()
    target_ind = i['target_ind'].cuda()
    image = i['image'].cuda() 
    
    mask_pred = model.forward(image)
    mask_pred = torch.sigmoid(mask_pred)

    losses = getDiceLoss(mask,mask_pred,target_ind,thresholds)
    if all_losses is None:
        all_losses=np.array(losses)
    else:
        all_losses=np.vstack((all_losses,np.array(losses)))
    
    target_ls.append(target_ind.item())
    
    # mask_pred = mask_pred.detach().cpu()
    # target_ind = target_ind.detach().cpu()
    # mask = mask.detach().cpu()

    # mask_pred = torch.gather(mask_pred,1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(640,640) 
    # mask = torch.gather(i['mask'],1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(640,640)

    #plt.imshow(mask.numpy())
    
    # 0.2,0.3,0.45,0.3,0.45


    # mask_pred = mask_pred.numpy()

    # cv2.imshow("predicted", ((mask_pred>0.1)*255/mask_pred.max()).astype('uint8'))
    # cv2.imshow("original", (mask.numpy()*255).astype('uint8'))
    # cv2.imshow("mask", mask.numpy())
    # cv2.imshow("predicted", mask_pred.numpy())

    # img2 = cv2.cvtColor(img.astype('int').astype('float32'), cv2.COLOR_RGB2BGR)
    # img2 = viz_img(img2 , res=(640,640), text=None, name="check", mask=mask.numpy())
    # cv2.imshow("check", img2.astype('uint8'))
#     cv2.waitKey()
# cv2.destroyAllWindows()
df_dict = {str(x): all_losses[:,k].reshape(-1).tolist() for k,x in enumerate(thresholds)}
df_dict['target'] = target_ls
loss_df = pd.DataFrame(df_dict)
print(loss_df.groupby('target').mean())
print(all_losses.shape)
print(np.mean(all_losses,axis=0))


# enable grids
#Simulate tissue thickness variations

# Add TTA for pred
# self teaching

# Downscale images per organ -- train and valid 

# add variable cutoff to dice selection during training  