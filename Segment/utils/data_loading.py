import logging
import os 
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import glob2

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, phase = 'train'):
        # self.images_dir = Path(images_dir)
        # print(self.images_dir)
        # self.masks_dir = Path(masks_dir)
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        # self.scale = scale
        # self.mask_suffix = mask_suffix

        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        # if not self.ids:
        #     raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        # logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.phase = phase
        self.classes = os.listdir(self.masks_dir + '/' + self.phase)
        self.mask_list = []
        self.img_list = []
        for clss in self.classes:
            mask_exist_list = os.listdir(self.masks_dir + '/' + self.phase + '/' + clss)
            for mask in mask_exist_list:
                img_path = self.images_dir + '/' + self.phase + '/' + clss + '/' + mask 
                mask_path = self.masks_dir + '/' + self.phase + '/' + clss + '/' + mask 
                if os.path.exists(img_path):
                    self.mask_list.append(mask_path)
                    self.img_list.append(img_path)
                else:
                    continue
        
    def __len__(self):
        return len(self.mask_list)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        return Image.open(filename)

    def __getitem__(self, idx):
        # name = self.ids[idx]
        # mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        # img_file = list(self.images_dir.glob(name + '.*'))

        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = self.load(self.mask_list[idx])
        img = self.load(self.img_list[idx])
        # print(mask.size)
        # mask = mask.convert("1")
        #mask.show()
        print(mask)
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class Diseasedata(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, phase='train')

#data = Diseasedata('/home/hqanh/Potato/Dataset/Segment/images', '/home/hqanh/Potato/Dataset/Segment/mask')
#img1 = data.__getitem__(10)
# print(img1)
