# -*- coding:utf-8 _*-
""" 
@author:limuyu
@file: train.py 
@time: 2019/05/13
@contact: limuyu0110@pku.edu.cn

"""

import random

import torch.optim as optim
from tqdm import tqdm

from config import *
from dataloader import diloader
from models.img_model import Generator, Discriminator
from models.text_model import TextEncoder
from utils import *

import numpy as np

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device('cuda:0')


def one_iter_discriminator(data, encoder, generator, discriminator, optimizerD, optimizerE, criterion, no_grad=0):
    real_img = data[0].to(device)
    seqs = data[1].to(device)
    lengths = data[2].to(device)
    seqs_fake = data[3].to(device)
    lengths_fake = data[4].to(device)

    _, vectors = encoder(seqs, lengths)
    vectors_new_shape = vectors.unsqueeze(2).unsqueeze(2)

    _, vectors_fake = encoder(seqs_fake, lengths_fake)

    discriminator.zero_grad()
    label = torch.full((real_img.size(0),), 1, device=device)
    output = discriminator(vectors, real_img)
    output = output.view(-1)
    errD_real = criterion(output, label)

    if not no_grad:
        errD_real.backward(retain_graph=True)
    optimizerE.step()

    D_x = output.mean().item()

    noise = torch.randn(vectors_new_shape.shape, device=device, std=0.1)

    fake = generator(vectors_new_shape, noise)
    label.fill_(0)
    output_1 = discriminator(vectors, fake.detach()).view(-1)
    errD_fake = criterion(output_1, label)

    if not no_grad:
        errD_fake.backward(retain_graph=True)

    output_2 = discriminator(vectors_fake, real_img).view(-1)
    errD_fake_2 = criterion(output_2, label)

    if not no_grad:
        errD_fake_2.backward()

    D_G_z1 = (output_1 + output_2).mean().item()
    optimizerD.step()

    return D_x, D_G_z1


def one_iter_generator(data, encoder, generator, discriminator, optimizerE, optimizerG, criterion):
    real_img = data[0].to(device)
    seqs = data[1].to(device)
    lengths = data[2].to(device)

    generator.zero_grad()
    encoder.zero_grad()
    label = torch.full((real_img.size(0),), 1, device=device)

    _, vectors = encoder(seqs, lengths)
    vectors_new_shape = vectors.unsqueeze(2).unsqueeze(2)
    noise = torch.randn(vectors_new_shape.shape, device=device, std=0.1)
    fake = generator(vectors_new_shape, noise)

    output = discriminator(vectors, fake).view(-1)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()

    optimizerG.step()
    optimizerE.step()

    return D_G_z2


def epoch_validation(test_loader, encoder, generator, discriminator, epoch, config):
    D_x, D_G_z1, D_G_z2 = 0, 0, 0

    for i, data in enumerate(test_loader):
        real_img = data[0].to(device)
        seqs = data[1].to(device)
        lengths = data[2].to(device)
        _, vectors = encoder(seqs, lengths)
        vectors_new_shape = vectors.unsqueeze(2).unsqueeze(2)

        output = discriminator(vectors, real_img).view(-1)
        D_x += output.mean().item()

        noise = torch.randn(vectors_new_shape.shape, device=device, std=0.1)
        fake = generator(vectors_new_shape, noise)
        output = discriminator(vectors, fake.detach()).view(-1)
        D_G_z1 += output.mean().item()

        output = discriminator(vectors, fake).view(-1)
        D_G_z2 += output.mean().item()

        validation_contrast_save(real_img, fake, epoch, config)

    print(
        F"[Validation] -- "
        F"D_x: {D_x / len(test_loader)}; "
        F"D_G_z1: {D_G_z1 / len(test_loader)}; "
        F"D_G_z2: {D_G_z2 / len(test_loader)};"
    )


def train(train_loader, test_loader, encoder, generator, discriminator, config):
    num_epochs = config['train']['num_epochs']
    num_iter_print_loss = config['train']['num_iter_print_loss']
    num_iter_check = config['train']['num_iter_check']

    lr = config['train']['lr']

    optimizerD = optim.Adam(discriminator.parameters(), lr=lr)
    optimizerG = optim.Adam(generator.parameters(), lr=lr)
    optimizerE = optim.Adam(encoder.parameters(), lr=lr)

    criterion = torch.nn.BCELoss()

    D_all_list = []

    for epoch in range(num_epochs):
        print(F"Begin Epoch[{epoch}]")
        D_check = np.zeros(3, dtype=float)
        D_epoch = np.zeros(3, dtype=float)
        D_print = np.zeros(3, dtype=float)

        update_D, update_G = 1, 1
        len_epoch = len(train_loader)

        for i, data in tqdm(enumerate(train_loader)):
            if update_D:
                tmp = one_iter_discriminator(data, encoder, generator, discriminator, optimizerD, optimizerE, criterion)
                D_check[:2] += tmp
                D_epoch[:2] += tmp
                D_print[:2] += tmp
            else:
                print('no')
                with torch.no_grad():
                    tmp = one_iter_discriminator(data, encoder, generator, discriminator, optimizerD, optimizerE, criterion, 1)
                    D_check[:2] += tmp
                    D_epoch[:2] += tmp
                    D_print[:2] += tmp

            tmp = one_iter_generator(data, encoder, generator, discriminator, optimizerE, optimizerG, criterion)
            D_check[2] += tmp
            D_epoch[2] += tmp
            D_print[2] += tmp

            if i and i % num_iter_print_loss == 0:
                print(
                    F"[{epoch}][{i}/{len(train_loader)}] -- "
                    F"D_x: {D_print[0] / num_iter_print_loss}; "
                    F"D_G_z1: {D_print[1] / num_iter_print_loss}; "
                    F"D_G_z2: {D_print[2] / num_iter_print_loss};"
                )
                D_print = np.zeros(3, dtype=float)

            if i and i % num_iter_check == 0:
                update_D = (D_check[0] / num_iter_check < config['train']['d_x_up']) \
                           or (D_check[1] / num_iter_check > config['train']['d_g_z1_down'])
                D_check = np.zeros(3, dtype=float)

        epoch_validation(test_loader, encoder, generator, discriminator, epoch, config)

        D_epoch /= len_epoch
        D_all_list.append(D_epoch)
        save_model(epoch, encoder, generator, discriminator, config['file'])
        plot_zag(D_all_list, config['file'])


if __name__ == '__main__':
    config = load_json(CONFIG_FILE)
    train_loader, test_loader = diloader.get_loader(config)
    encoder = TextEncoder(config['text'])
    encoder = encoder.to(device)
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    weights_init(generator)
    weights_init(discriminator)
    print_params_num(generator, discriminator, encoder)
    train(train_loader, test_loader, encoder, generator, discriminator, config)
