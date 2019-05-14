# -*- coding:utf-8 _*-
""" 
@author:limuyu
@file: utils.py 
@time: 2019/05/13
@contact: limuyu0110@pku.edu.cn

"""

import os
import torch
import torchvision.utils as tvutils
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
import json
import codecs

import torch.nn as nn

def get_fl(path):
    return [os.path.join(path, fn) for fn in os.listdir(path)]


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def validation_contrast(real, fake):
    real = tvutils.make_grid(real.cpu(), 2).detach().numpy()
    fake = tvutils.make_grid(fake.cpu(), 2).detach().numpy()

    plt.subplot(2, 1, 1)
    plt.imshow(np.transpose(real, (1, 2, 0)))

    plt.subplot(2, 1, 2)
    plt.imshow(np.transpose(fake, (1, 2, 0)))

    plt.show()



def load_json(path):
    with codecs.open(path, 'r', 'utf8') as f:
        return json.load(f)


def dump_json(obj, path):
    with codecs.open(path, 'w', 'utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

