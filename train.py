import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt

from data_loader import *
import pandas as pd

from visualization import viz_img

from train_handler import TrainHandler

from segformer import segformersegmentation, segformersegmentationmitb3

from PVT import *

# from PVT import *
seed = 123
set_seed(seed)

df_train = pd.read_csv(TRAIN_CSV_FILE)
df_train['kfold'] = 1
df_train.loc[df_train.sample(frac = 0.15, random_state=seed).index.values,'kfold'] = 0

config = {
'batch_size': 4,
'evaluate_interval': 1,
'epochs': 200,
'num_folds': 1,
'scheduler': 'onecycle',
'loss': 'symmetric_lovasz',
'metric': 'dice',
}

train_loader, valid_loader = make_loader(df_train, config['batch_size'], (640,640), seed=seed)

config['log_interval'] = len(train_loader)

model = SemanticFPN_PVT(backbone_model = "pvt_v2_b4", mode='train', size=640, num_classes=5, pt_weights_dir = "model/pvt_v2_b4.pth")

# model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_DIR,"pt_b3",  "model0.bin")), strict=True)

# for layer in model.backbone.parameters():
#     layer.require_grad=False

# 
# model = segformersegmentation(mode="train", size=640)


# model = smp.UnetPlusPlus(
#     encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=5,                      # model output channels (number of classes in your dataset)
# )

#model.load_state_dict(torch.load(os.path.join( MODEL_OUTPUT_DIR ,"baseline+lrsched", "model0.bin")), strict=True)

if torch.cuda.device_count() >= 1:
    print('Model pushed to {} GPU(s), type {}.'.format(
        torch.cuda.device_count(),
        torch.cuda.get_device_name(0))
    )
    model = model.cuda()
else:
    raise ValueError('CPU training is not supported')

# optimizer = torch.optim.Adam([
#         {'params': model.decoder.parameters(), 'lr': 1e-3},
#         {'params': model.encoder.parameters(), 'lr': 1e-3},
#     ])

optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 3e-5},
    ])

num_steps = len(train_loader)*config['epochs']
if config['scheduler']=='multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(num_steps/3), int(num_steps*2/3)], gamma=0.6, last_epoch=- 1, verbose=False)
elif config['scheduler'] == 'onecycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr =1e-4, epochs=config['epochs'], steps_per_epoch=len(train_loader), pct_start=0.5, anneal_strategy='cos', div_factor=5, final_div_factor=10 ,cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, last_epoch=- 1)
trainHandler = TrainHandler(model, train_loader, valid_loader, optimizer, scheduler, config)
trainHandler.run()

#  model

# x = torch.rand(1,3,640,640)
# mask = torch.rand(1,1,64,64)

# mask_y = model.forward(x)

# print(mask_y.shape)


# for i in train_loader:
#     img = i['image'][0,:,:,:].view(3,640,640).permute((1, 2, 0)).numpy()
#     print(i['target_ind'])
#     mask_shape = i['mask'].shape
    
#     mask = torch.gather(i['mask'],1, i['target_ind'].view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))[0,:,:,:].view(1,640,640)

#     img2 = cv2.cvtColor(img.astype('int').astype('float32'), cv2.COLOR_RGB2BGR)
#     img2 = viz_img(img2 , res=(640,640), text=None, name="check", mask=mask.numpy())
#     cv2.imshow("check", img2.astype('uint8'))
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     break
