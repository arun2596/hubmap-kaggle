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
            self.data_dir = TRAIN_IMAGES_DIR
        elif self.mode == 'test':
            self.data_dir = TEST_IMAGES_DIR
        else:
            raise Exception("Invalid mode: " + str(self.mode))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_id, rle = self.img_id[item], self.rles[item]

        img_name = os.path.join(self.data_dir, str(img_id) + '.tiff')

        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        
        mask = np.zeros((max(self.class_id), image.shape[0], image.shape[1]))
        mask[self.class_id[item]-1,:,:]  = rleToMask(rle,image.shape[0],image.shape[1])

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
        input_shape=(640, 640),
        fold=0,
):
    dataset = {'train': data[data['kfold'] != fold], 'valid': data[data['kfold'] == fold]}
    

    transform = {'train': transforms.Compose([
        SquarePad(),
        Rescale(input_shape),
        Rerange(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        FlipLR(0.5)
    ]),
        'valid': transforms.Compose([
            SquarePad(),
            Rescale(input_shape),
            Rerange(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])}

    train_dataset, valid_dataset = [
        DatasetRetriever(dataset[mode], transform=transform[mode], mode=mode) for mode in
        ['train', 'valid']]

    train_sampler = RandomSampler(dataset['train'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    valid_sampler = SequentialSampler(dataset['valid'])
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size // 2,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    return train_loader, valid_loader