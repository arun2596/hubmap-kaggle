from transformers import SegformerModel, SegformerConfig, SegformerFeatureExtractor, SegformerForSemanticSegmentation
import json
import torch


class segformersegmentation(torch.nn.Module):
    def __init__(self, mode='train', size=640, config_json=None):
        super().__init__()
        self.mode=mode
        self.size=size
        if self.mode=="test":
            with open(config_json,"r") as file:
                config_dict = json.load(config_json)    
                config = SegformerConfig(**config_dict)
                self.segpretrained = SegformerForSemanticSegmentation(config)
        else:
            self.segpretrained = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
            self.upsample=torch.nn.Upsample(size=self.size, mode='bilinear', align_corners=False)
        self.segpretrained.decode_head.classifier= torch.nn.Conv2d(768, 5, kernel_size=(1, 1), stride=(1, 1))
 
       

    def forward(self, x):
        out_logits =  self.segpretrained.forward(x).logits
        if self.mode!='test':
            out_logits = self.upsample(out_logits)
        return out_logits

# model = segformersegmentation()
# x = torch.rand((2,3,640,640))
# print(model.forward(x).shape)