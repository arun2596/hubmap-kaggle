import torch 

import cv2
import os
from torchvision import transforms
import torch
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

from config import *
from model.utils import *

# from utils.utils import SquarePad, Rescale, Normalize, Rerange, FlipLR


class DatasetRetriever(Dataset):
    def __init__(self, data, mode='train', transform=None):
        self.data = data

        self.img_id = self.data.id.values.tolist()
        self.rles = self.data.rle.values.tolist()
        self.classes = self.data.organ.values.tolist()
        self.class_id = [CLASS_TO_ID[x] for x in self.classes]
        self.mode = mode

        if self.mode == 'train' or self.mode == 'valid':
            self.data_dir = TRAIN_IMAGES_DIR_640
        elif self.mode == 'test':
            self.data_dir = TEST_IMAGES_DIR
        else:
            raise Exception("Invalid mode: " + str(self.mode))

        self.mask_data_dir = TRAIN_MASK_DIR_640
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_id, rle = self.img_id[item], self.rles[item]

        #adding stained images
        stain_prob=0.20
        stain_flag=False
        if random.random()<stain_prob and self.mode=='train':
            stain_flag=True
            img_name = os.path.join(STAINED_IMAGES_DIR, str(img_id) + '.tiff')
        else:
            img_name = os.path.join(self.data_dir, str(img_id) + '.tiff')

        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        if stain_flag:
            image= cv2.resize(image,(640,640))

        mask_layer = cv2.imread(os.path.join(self.mask_data_dir, str(img_id)+ "_mask.tiff"))[:,:,0]
        
        mask = np.zeros((max(self.class_id), image.shape[0], image.shape[1]))
        mask[self.class_id[item]-1,:,:]  = mask_layer.reshape(mask_layer.shape[0], mask_layer.shape[1])

        target_ind = self.class_id[item]-1


        # NOTE: Images is transposed from (H, W, C) to (C, H, W)
        sample = {
            'image': torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'target_ind': torch.tensor(target_ind, dtype=torch.int64),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def make_loader(
        data,
        batch_size,
        input_shape=640,
        fold=0,
        seed=123
):
    dataset = {'train': data[data['kfold'] != fold], 'valid': data[data['kfold'] == fold]}
    

    transform = {'train': transforms.Compose([
        # FlipLR(0.3),
        # FlipTD(0.3),
        ToAlbuNumpy(),
        AlbuHSVShift(0.4),
        AlbuBrightnessContrast(0.2),
        AlbuDownScale(0.15, input_shape=input_shape),
        AlbuAngleRotate(0.4),
        AlbuElastic(0.3),
        ToTensor(),
        Rerange(),
        Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
        
    ]),
        'valid': transforms.Compose([
            # SquarePad(),
            ToAlbuNumpy(),
            AlbuDownScale(1, input_shape=input_shape,mode='valid'),
            ToTensor(),
            Rerange(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])}

    train_dataset, valid_dataset = [
        DatasetRetriever(dataset[mode], transform=transform[mode], mode=mode) for mode in
        ['train', 'valid']]

    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_val =torch.Generator()
    g_val.manual_seed(seed)
    train_sampler = RandomSampler(dataset['train'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=16
        # worker_init_fn=seed_worker,
        # generator=g_train
    )

    valid_sampler = SequentialSampler(dataset['valid'])
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size // 2,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=2
        # worker_init_fn=seed_worker,
        # generator=g_val

    )

    return train_loader, valid_loader