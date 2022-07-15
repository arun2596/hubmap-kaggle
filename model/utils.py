
import numpy as np
from itertools import groupby

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


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

        p_left, p_top = [(max_wh - s) // 2 for s in img_size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(img_size, [p_left, p_top])]
        padding = [p_left, p_top, p_right, p_bottom]
        return {'image': F.pad(image, padding, 0, 'constant'), 'mask': F.pad(mask, padding, 0, 'constant'), 'target_ind':target_ind}


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


################ LOSS FUNCITONS #############



def DiceLoss(mask, mask_pred, target_ind, smooth=1):
    #
    mask_shape = mask.shape
    mask_pred = torch.gather(mask_pred,1, target_ind.view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))
    mask = torch.gather(mask,1, target_ind.view(-1,1,1,1).repeat(1,1,mask_shape[-2],mask_shape[-1]))
    
    #flatten
    mask = mask.view(mask.shape[0], -1)
    mask_pred = mask_pred.view(mask_pred.shape[0],-1)
    #element_wise production to get intersection score
    intersection = (mask*mask_pred).sum(dim=1)

    dice_score = torch.div(2*intersection + smooth , mask.sum(dim=1) + mask_pred.sum(dim=1) + smooth)
    return 1 - dice_score.mean()