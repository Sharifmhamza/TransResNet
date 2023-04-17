#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from transresnet import TRANSRESNET


class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Dataset.Config(datapath=path, snapshot=model, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                mask  = mask.cuda().float()
                p = self.net(image, shape=shape)
                pred   = F.interpolate(p[0],size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred[torch.where(pred>0)] /= (pred>0).float().mean()
                pred[torch.where(pred<0)] /= (pred<0).float().mean()
                pred   = torch.sigmoid(pred)
                pred  = (pred*255).cpu().numpy()
                head  = '../src/result/'+self.model+'/'
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
#278
if __name__=='__main__':
    for path in ['../src/data/polyp/test/CVC-ClinicDB']:
        for model in ['model-267']:
                t = Test(dataset,TRANSRESNET, path,'../src/polyp-model-p15-LR/'+model)
                t.save()