#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class random_image_enhance(object):
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, image, mask):
        #image = Image.fromarray(image)
        np.random.shuffle(self.enhance_method)

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(image)
                factor = float(1 + np.random.random() / 10)
                image = enhancer.enhance(factor)
        image = image

        return image, mask


class random_gaussian_blur(object):
    def __init__(self):
        pass

    def __call__(self, image, mask):
        image = image
        if np.random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        image = image

        return image, mask

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask):
        image = (image - self.mean)/self.std
        mask /= 255
        return image, mask


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1]
        else:
            return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class Transformation(object):

    def __init__(self):
        self.transforms =  A.Compose([
            #A.UnsharpMask(p=0.5),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.RandomContrast(p=0.3),
            A.ShiftScaleRotate(p=0.5),
            A.Blur(p=0.3),
            A.LongestMaxSize(1024, p=1),
            #A.MaskDropout(p=0.5),
            #A.ColorJitter(p=0.2),
            #A.Emboss(p=0.2),
            ])

    def __call__(self, image, mask):
        #image = np.array(image)
        #mask  = np.array(mask)
        transform = self.transforms(image=image, mask=mask)
        return transform["image"], transform["mask"]

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[0.485, 0.456, 0.406]]])
        self.std    = np.array([[[0.229, 0.224, 0.225]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.samples    = []
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        #self.randomflip = RandomFlip()
        self.resize     = Resize(1024, 1024)
        self.randomcrop = RandomCrop()
        self.totensor   = ToTensor()
        self.guassian   = random_gaussian_blur()
        self.image_root = cfg.kwargs['datapath'] +'/images/'
        self.mask_root  = cfg.kwargs['datapath'] +'/masks/'
        image_lst = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        for each in image_lst:
            img_name = each.split("/")[-1]
            img_name = img_name.split(".")[0]
            self.samples.append(img_name)


    def __getitem__(self, idx):
        
        name  = self.samples[idx]
        image = cv2.imread(self.image_root + name+'.png').astype(np.float32)
        image = image[:,:,::-1].copy()
        mask = cv2.imread(self.mask_root  + name+'.png', 0).astype(np.float32)
        shape = mask.shape
        #print(shape)

        if self.cfg.mode=='train':
            #image, mask = self.resize(image, mask)
            image, mask = self.normalize(image, mask)
            #image, mask = self.enhance(image, mask)
            #image, mask = self.dilation(image, mask)
            #image, mask = self.guassian(image, mask)
            #image, mask = self.randomflip(image, mask)
            #image, mask = self.transform(image, mask)
            #image, mask = self.randomcrop(image, mask)
            return image.copy(), mask.copy()
        else:
            shape = mask.shape #
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

    def collate(self, batch):
        size = 1024
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask

    def __len__(self):
        return len(self.samples)

