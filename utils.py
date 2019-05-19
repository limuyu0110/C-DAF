# -*- coding:utf-8 _*-
""" 
@author:limuyu
@file: utils.py 
@time: 2019/05/13
@contact: limuyu0110@pku.edu.cn

"""

import os
import torch
from torch.nn import Module
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


def validation_contrast_save(real, fake, epoch):
    img = tvutils.make_grid(torch.cat([real, fake], dim=0).cpu(), 2).detach().numpy()
    plt.imsave('./results/' + F'{epoch}.jpg', np.transpose(img, (1, 2, 0)))



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


def plot_zag(D_xs, D_G_z1s, D_G_z2s):

    with open('./results/stat.json', 'w') as f:
        json.dump({
            'D_x': D_xs,
            'D_G_z1': D_G_z1s,
            'D_G_z2': D_G_z2s,
        }, f, ensure_ascii=False, indent=4
        )
    plt.figure()
    plt.title('Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.plot(D_xs, color='green', label='D_xs')
    plt.plot(D_G_z1s, color='red', label='D_G_z1')
    plt.plot(D_G_z2s, color='blue', label='D_G_z2')
    # plt.plot()
    plt.legend()
    plt.savefig('results/gram.png')
    # plt.show()


def save_model(epoch, encoder, generator, discriminator):
    file = F"checkpoints/{epoch}.pt"
    d = {
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }
    torch.save(d, file)


def load_model(epoch, encoder: Module, generator: Module, discriminator: Module):
    file = F"checkpoints/{epoch}.pt"
    d = torch.load(file)

    encoder.load_state_dict(d['encoder'])
    generator.load_state_dict(d['generator'])
    discriminator.load_state_dict(d['discriminator'])

    return encoder, generator, discriminator


if __name__ == '__main__':
    plot_zag([2, 1, 3], [1, 2, 3], [3, 2, 1])
