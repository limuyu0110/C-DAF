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
    return json.load(open(path, 'r'))

