
import pv2_models
import torch
from torch import nn
import torch.nn.functional as F
from daformer import daformer_conv3x3

# model_config = Config.fromfile("fpn_pvtv2_b2_ade20k_40k.py")
# model = build_segmentor(model_config.model)
# model = model.cuda()


class SemanticFPN_PVT(torch.nn.Module):
    def __init__(self, backbone_model = "pvt_v2_b2", mode='train', size=640, num_classes=5, pt_weights_dir = "model/pvt_v2_b2.pth"):
        super().__init__()
        self.mode=mode
        self.size=size

        self.num_classes=num_classes

        self.backbone_model_name = backbone_model

        self.backbone = getattr(pv2_models, self.backbone_model_name)()
        if mode=='train':
            self.backbone.load_state_dict(torch.load(pt_weights_dir), strict=False)
        # LOAD THE PRETRAINED BACKBONE HERE IF ITS IN TRAIN MODE
        
        # Top layer
        self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # Semantic branch
        self.semantic_branch = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(32, 32) 
        self.gn2 = nn.GroupNorm(64, 64)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):
        # Bottom-up using backbone
        low_level_features = self.backbone(x)

        c2 = low_level_features[0]
        c3 = low_level_features[1]
        c4 = low_level_features[2]
        c5 = low_level_features[3]

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))


        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)


        # Semantic
        _, _, h, w = p2.size()
        # 64->64
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 64->64
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # 64->32
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 64->64
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 64->32
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 64->32
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        return self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)


class DaformerFPN_PVT(torch.nn.Module):
    def __init__(self, backbone_model = "pvt_v2_b4", mode='train', size=640, num_classes=5, pt_weights_dir = "model/pvt_v2_b4.pth", decoder=daformer_conv3x3, decoder_dim=256):
        super().__init__()
        self.mode=mode
        self.size=size
        self.decoder_dim=decoder_dim
        self.num_classes=num_classes

        self.backbone_model_name = backbone_model
        self.decoder = decoder(encoder_dim=[64,128,320,512], decoder_dim=self.decoder_dim)
        self.backbone = getattr(pv2_models, self.backbone_model_name)()
        if mode=='train':
            self.backbone.load_state_dict(torch.load(pt_weights_dir), strict=False)
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



# model = DaformerFPN_PVT()
# model = model.cuda()
# x = torch.rand((1,3,640,640))
# x = x.cuda()
# print(model.forward(x).shape)