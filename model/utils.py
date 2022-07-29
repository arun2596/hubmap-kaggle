
import numpy as np
from itertools import groupby
import staintools
from config import TEST_IMAGES_DIR
import os 

def rleToMask(rleString,height,width):
  rows,cols = height,width
  rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
  rlePairs = np.array(rleNumbers).reshape(-1,2)
  img = np.zeros(rows*cols,dtype=np.uint8)
  for index,length in rlePairs:
    index -= 1
    img[index:index+length] = 1
  img = img.reshape(cols,rows).T
  return img

def maskToRle(binary_mask):
    rle=""
    ind= 0
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
      if value == 0:
        ind+=len(list(elements))
      else:
        rle+=" "
        count = len(list(elements))
        rle+=str(ind+1) + " " + str(count)
        ind+=count
    return rle.strip()


####### DATA UTILS ############

import torch
import torchvision
import torchvision.transforms.functional as F
import torch.nn as nn
import numpy as np
import random

from sklearn import model_selection

import albumentations as A

import cv2


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return



def create_folds(data, num_splits, seed, cross_validation=True):
    if cross_validation:
        data["kfold"] = -1
        kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=seed)
        for f, (t_, v_) in enumerate(kf.split(X=data)):
            data.loc[v_, 'kfold'] = f
    else:
        print('splitting data with validation set  15%')
        data["kfold"] = np.random.choice([-1,0], size=data.shape[0], replace=True, p=[0.85, 0.15])

    return data


class Logger():
    def __init__(self, output_location):
        self.body = []
        self.header = []
        self.footer = []
        self.output_location = output_location

    def log(self, text, place='body'):
        if place == 'header':
            self.header.append(text)
        elif place == 'footer':
            self.footer.append(text)
        elif place == 'body':
            self.body.append(text)

        print(text)
        return

    def save_log(self):
        with open(self.output_location, 'a+') as the_file:
            for line in self.header:
                the_file.write(str(line) + '\n')
            the_file.write('-' * 50)
            the_file.write('\n')
            for line in self.body:
                the_file.write(str(line) + '\n')
            the_file.write('-' * 50)
            the_file.write('\n')
            for line in self.footer:
                the_file.write(str(line) + '\n')
            the_file.write('-' * 50)
            the_file.write('\n')
        return


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val




class SquarePad(object):
    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']

        img_size = image.size()[1:]
        max_wh = max(img_size)

        p_top, p_left = [(max_wh - s) // 2 for s in img_size]
        p_bottom, p_right = [max_wh - (s + pad) for s, pad in zip(img_size, [p_top, p_left])]
        padding = [p_left, p_top, p_right, p_bottom]
        return {'image': F.pad(image, padding, 0, 'constant'), 'mask': F.pad(mask, padding, 0, 'constant'), 'target_ind':target_ind}

class PadToSize(object):
    def __init__(self, size, proba):
        self.size = size
        self.proba=proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']

        img_size = image.size()[1:]
        max_wh = max(img_size)
        if max_wh<self.size and random.random()<self.proba:
            p_top, p_left = [(self.size - s) // 2 for s in img_size]
            p_bottom, p_right = [self.size - (s + pad) for s, pad in zip(img_size, [p_top, p_left])]
            padding = [p_left, p_top, p_right, p_bottom]
            
            image =  F.pad(image, padding, 0, 'constant')
            mask = F.pad(mask, padding, 0, 'constant')
        return {'image': image, 'mask': mask, 'target_ind':target_ind}

class StainNormalise(object):
    def __init__(self, proba):
        self.proba=proba
        self.normalizer = staintools.StainNormalizer(method='vahadane')

        img = cv2.cvtColor(cv2.imread(os.path.join(TEST_IMAGES_DIR,"10078.tiff")), cv2.COLOR_BGR2RGB)

        self.normalizer.fit(img)


    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        if random.random()<self.proba:
            image = self.normalizer.transform(image)
  
        return {'image': image, 'mask': mask, 'target_ind':target_ind}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']

        h, w = image.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = F.resize(image, (new_h, new_w))
        mask = F.resize(mask, (new_h, new_w))
        mask = torch.where(mask>0,1,0)
        return {'image': img, 'mask': mask, 'target_ind': target_ind}

class Rerange(object):

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']

        return {'image': image / 255.0, 'mask': mask, 'target_ind': target_ind}


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']

        img = F.normalize(image, mean=self.mean, std=self.std)

        return {'image': img, 'mask': mask, 'target_ind': target_ind}


class FlipLR(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        if random.random()<self.proba:
          image = F.hflip(image)
          mask = F.hflip(mask)
        
        

        return {'image': image, 'mask': mask, 'target_ind': target_ind}

class FlipTD(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        if random.random()<self.proba:
          image = F.vflip(image)
          mask = F.vflip(mask)

        return {'image': image, 'mask': mask, 'target_ind': target_ind}


class AlbuHSVShift(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        image=  A.augmentations.transforms.HueSaturationValue(p=self.proba, hue_shift_limit=180, sat_shift_limit=30, val_shift_limit=20)(image=image)['image']
        return {'image': image, 'mask': mask, 'target_ind': target_ind}

class AlbuRandomScale(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        albu_res =  A.augmentations.geometric.resize.RandomScale(scale_limit=(-0.9,0.5), interpolation=1, always_apply=False, p=self.proba)(image=image, mask= mask[target_ind,:,:])
        image = albu_res['image']
        mask = np.zeros((mask.shape[0],albu_res['mask'].shape[0],albu_res['mask'].shape[1]))
        mask[target_ind,:,:] = (albu_res['mask']>0)*1
        return {'image': image, 'mask': mask, 'target_ind': target_ind}


class AlbuAngleRotate(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        albu_out =  A.augmentations.geometric.rotate.Rotate (limit=180, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, crop_border=False, always_apply=False, p=self.proba)(image=image, mask= mask[target_ind,:,:])
        image, mask[target_ind,:,:] = albu_out['image'], albu_out['mask']
        return {'image': image, 'mask': mask, 'target_ind': target_ind}

class AlbuElastic(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        albu_out =  A.augmentations.geometric.transforms.ElasticTransform (alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=False, approximate=False, same_dxdy=False, p=self.proba)(image=image, mask= mask[target_ind,:,:])
        image, mask[target_ind,:,:] = albu_out['image'], albu_out['mask']
        return {'image': image, 'mask': mask, 'target_ind': target_ind}

class AlbuCoarseDropout(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        albu_out =  A.augmentations.dropout.coarse_dropout.CoarseDropout (max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=0, always_apply=False, p=self.proba)(image=image, mask= mask[target_ind,:,:])
        image, mask[target_ind,:,:] = albu_out['image'], albu_out['mask']
        return {'image': image, 'mask': mask, 'target_ind': target_ind}


class AlbuDownScale(object):

    def __init__(self,proba, input_shape=640, mode='train'):
        self.proba=proba
        self.input_shape = input_shape
        self.mode=mode
    def __call__(self,sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        scale_min=0.2
        scale_max=0.7

        if self.mode=='valid':
            test_px = {
                    0:6.263,
                    1:0.4945,
                    2:0.7562,
                    3:0.5,
                    4:0.229,
                }
            if isinstance(self.input_shape, int):
                pass
            else:
                input_shape = self.input_shape[0]
            scale = (3000*0.4)/(test_px[int(target_ind)]*input_shape)
            if scale>=1:
                scale=1
            scale_min = scale_max = scale
            
        if scale_min!=1:
            image = A.augmentations.transforms.Downscale (scale_min=scale_min, scale_max=scale_max, interpolation=0, always_apply=False, p=self.proba)(image=image)['image']
        return {'image': image, 'mask': mask, 'target_ind': target_ind}

class AlbuBrightnessContrast(object):

    def __init__(self,proba):
        self.proba=proba
    
    def __call__(self,sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        image = A.augmentations.transforms.RandomBrightnessContrast (brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=self.proba)(image=image)['image']
        return {'image': image, 'mask': mask, 'target_ind': target_ind}



class ToAlbuNumpy():
    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        sample = {
            'image': image.numpy().astype('uint8').transpose((1,2,0)),
            'mask': mask.numpy(),
            'target_ind': target_ind.numpy(),
        }
        return sample  

class ToTensor():
    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask, target_ind = sample['image'], sample['mask'], sample['target_ind']
        sample = {
            'image': torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'target_ind': torch.tensor(target_ind, dtype=torch.int64),
        }
        return sample   


################ LOSS FUNCITONS #############

from lovazs_loss import lovasz_hinge


def DiceLoss(mask, mask_pred, target_ind, with_logits=False, smooth=1, cutoff=None):
    #
    if with_logits:
        mask_pred = torch.sigmoid(mask_pred)    
    mask_shape = mask.shape
    mask_pred = torch.gather(mask_pred,1, target_ind.view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))
    mask = torch.gather(mask,1, target_ind.view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))
    
    if cutoff!=None:
        mask_pred = torch.where(mask_pred>cutoff,1,0)

    #flatten
    mask = mask.view(mask.shape[0], -1)
    mask_pred = mask_pred.view(mask_pred.shape[0],-1)
    #element_wise production to get intersection score
    intersection = (mask*mask_pred).sum(dim=1)

    dice_score = torch.div(2*intersection + smooth , mask.sum(dim=1) + mask_pred.sum(dim=1) + smooth)
    return 1 - dice_score.mean()

def symmetric_lovasz(mask, mask_pred, target_ind):
    mask_shape = mask.shape
    mask_pred = torch.gather(mask_pred,1, target_ind.view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))
    mask = torch.gather(mask,1, target_ind.view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))
    return 0.5*(lovasz_hinge(mask_pred, mask) + lovasz_hinge(-mask_pred, 1.0 - mask))