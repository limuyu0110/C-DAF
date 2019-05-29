#-*- coding:utf-8 _*-  
""" 
@author:limuyu
@file: img_model.py 
@time: 2019/05/13
@contact: limuyu0110@pku.edu.cn

"""

import torch
import torch.nn as nn
from overrides import overrides


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        nz = config['text']['hidden']
        ngf = config['generator']['features']
        nc = config['img']['channels']
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #
            # nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input, noise):
        a = torch.cat([input, noise], dim=1)
        return self.main(a)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        ndf = config['discriminator']['features']
        self.n = ndf
        nc = config['img']['channels']
        self.batch_size = config['train']['batch_size']
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf, 4, 1, 0, bias=False),

        )
        self.linear_final = nn.Linear(ndf + config['text']['hidden'], 1)
        self.image_features = config['discriminator']['features']
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_vector, image):
        image_feature = self.main(image)
        image_feature = image_feature.view(-1, self.image_features)
        a = torch.cat([text_vector, image_feature], dim=1)
        a = self.linear_final(a)
        a = self.sigmoid(a)
        return a

