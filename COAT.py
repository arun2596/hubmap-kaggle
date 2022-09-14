

import coat_models
import torch
from torch import nn
import torch.nn.functional as F
from daformer import daformer_conv3x3


class DaformerFPN_COAT(torch.nn.Module):
    def __init__(self, backbone_model = "coat_lite_medium", mode='train', size=640, num_classes=5, pt_weights_dir = "model/coat_lite_medium_384x384_f9129688.pth", decoder=daformer_conv3x3, decoder_dim=320, encoder_dim = [128,256,320,512], load_strict=False):
        super().__init__()
        self.mode=mode
        self.size=size
        self.decoder_dim=decoder_dim
        self.num_classes=num_classes
        self.encoder_dim = encoder_dim
        self.backbone_model_name = backbone_model
        self.decoder = decoder(encoder_dim=self.encoder_dim, decoder_dim=self.decoder_dim)
        self.backbone = getattr(coat_models, self.backbone_model_name)(return_interm_layers=True)
        if mode=='train':
            self.backbone.load_state_dict(torch.load(pt_weights_dir)['model'], strict=load_strict)
        # LOAD THE PRETRAINED BACKBONE HERE IF ITS IN TRAIN MODE
        
        self.final_conv = nn.Conv2d(self.decoder_dim, self.num_classes,1)
    
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)



    def forward(self, x):
        # Bottom-up using backbone
        low_level_features = self.backbone(x)
        df_out, decoder_out = self.decoder(low_level_features)
        h,w = df_out.shape[-2:]
        return self._upsample(self.final_conv(df_out), 4 * h, 4 * w)

if 0:
    a = DaformerFPN_COAT()
    x = torch.rand((1,3,768,768))
    print(a.forward(x).shape)