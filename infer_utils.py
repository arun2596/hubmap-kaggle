from model.utils import DiceLoss
import torch

def getDiceLoss(mask,mask_pred, target_ind,thresholds):
    loss = []
    for threshold in thresholds:
        loss.append(DiceLoss(mask, torch.where(mask_pred>threshold,1,0), target_ind).item())

    return loss