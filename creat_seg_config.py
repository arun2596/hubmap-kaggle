from transformers import SegformerModel, SegformerConfig, SegformerFeatureExtractor, SegformerForSemanticSegmentation
import json
import torch
import json

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3")

# print(type(eval(model.config.__str__()[16:].replace('true','True'))))

with open("model/mit_b3_config.json", "w") as file:
    json.dump(eval(model.config.__str__()[16:].replace('true','True')), file)