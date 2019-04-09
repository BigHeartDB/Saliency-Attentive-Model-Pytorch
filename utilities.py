from __future__ import division
import cv2
import numpy as np
import scipy.io
import scipy.ndimage
from torch.utils.data import Dataset
from PIL import Image
import torch
from scipy.io import loadmat
from config import shape_r_out, shape_c_out, b_s
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# complement the class of Mydataset
class MyDataset(Dataset):
    def __init__(self, imgs_txt_path, maps_txt_path,  fixs_txt_path,
                 transform_img = None, transform_map = None, transform_fix = None):
        fh_imgs = open(imgs_txt_path, 'r')
        fh_maps = open(maps_txt_path, 'r')
        fh_fixs = open(fixs_txt_path, 'r')
        imgs = []
        maps = []
        fixs = []
        for line in fh_imgs:
            line = line.rstrip()
            imgs.append(line)
        for line in fh_maps:
            line = line.rstrip()
            maps.append(line)
        for line in fh_fixs:
            line = line.rstrip()
            fixs.append(line)

        self.imgs = imgs
        self.maps = maps
        self.fixs = fixs
        self.transform_img = transform_img
        self.transform_map = transform_map
        self.transform_fix = transform_fix

    def __getitem__(self, index):
        img_path_cur = self.imgs[index]
        map_path_cur = self.maps[index]
        fix_path_cur = self.fixs[index]
        img = Image.open(img_path_cur).convert('RGB')
        map = Image.open(map_path_cur)
        fix = loadmat(fix_path_cur)
        fix = fix.get('fixLocs')
        fix = transforms.ToPILImage()(fix)

        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_map is not None:
            map = self.transform_map(map)
        if self.transform_fix is not None:
            fix = self.transform_fix(fix)

        return img, map, fix

    def __len__(self):
        return len(self.imgs)


# add temporal dimention
def format_attLSTM(x, nb_ts):

    y = []
    x.unsqueeze_(1)
    for i in range(nb_ts):
        y.append(x)
    feature = torch.cat(y,1)

    return feature


# preprocessing the fixs dataset
def fixs_preprocessing(fixs):

    fixs_out = []
    for i in range(b_s):
       img = transforms.ToPILImage()(fixs[i])
       plt.imshow(img)
       plt.show()

    print('Done!')