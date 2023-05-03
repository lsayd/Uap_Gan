#-*- coding = utf-8 -*-
#@Time : 2023/3/10 下午3:39
#@Author : lv
#@File : processImage.py
#@Software : PyCharm
from model import Generator, AvgBlurGenerator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import attacks

from PIL import ImageFilter
from PIL import Image
from torchvision import transforms


from universal_pert import universal
from data_loader import get_loader

def create_labels( c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                # Reverse attribute value.
                c_trg[:, i] = (c_trg[:, i] == 0)


        c_trg_list.append(c_trg)
    return c_trg_list

def ProcessImage(data_loader):

    for i, (x_real, c_org) in enumerate(data_loader):

        x_arr = np.array(x_real.cpu())
        file_perturbation = os.path.join('data', 'universal.npy')
        v = np.load(file_perturbation)
        v = torch.tensor(v)
        x_adv = x_real+v*5
        out = (x_adv + 1) / 2
        save_dir = 'UniversalImage'
        result_path = os.path.join(
            save_dir, '{}-images.jpg'.format(i + 1))
        save_image(out, result_path)
        if i == 99:  # stop after this many images
            break


