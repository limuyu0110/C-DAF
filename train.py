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

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device('cuda:0')


def one_iter(data, encoder, generator, discriminator, optimizerE, optimizerG, optimizerD, criterion):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    real_img = data[0].to(device)
    seqs = data[1].to(device)
    lengths = data[2].to(device)

    _, vectors = encoder(seqs, lengths)
    vectors = vectors.unsqueeze(2).unsqueeze(2)

    discriminator.zero_grad()
    b_size = real_img.size(0)
    label = torch.full((b_size,), 1, device=device)
    output = discriminator(real_img)
    output = output.view(-1)
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    fake = generator(vectors)
    label.fill_(0)
    output = discriminator(fake.detach()).view(-1)
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    generator.zero_grad()
    label.fill_(1)
    output = discriminator(fake).view(-1)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()
    optimizerE.step()

    return errD.mean().item(), errG.mean().item(), D_x, D_G_z1, D_G_z2


def one_iter_discriminator(data, encoder, generator, discriminator, optimizerD, criterion, no_grad=0):
    real_img = data[0].to(device)
    seqs = data[1].to(device)
    lengths = data[2].to(device)

    _, vectors = encoder(seqs, lengths)
    vectors = vectors.unsqueeze(2).unsqueeze(2)

    discriminator.zero_grad()
    label = torch.full((real_img.size(0),), 1, device=device)
    output = discriminator(real_img)
    output = output.view(-1)
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    if not no_grad:
        errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    fake = generator(vectors)
    label.fill_(0)
    output = discriminator(fake.detach()).view(-1)
    errD_fake = criterion(output, label)
    if not no_grad:
        errD_fake.backward()
    D_G_z1 = output.mean().item()
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
    vectors = vectors.unsqueeze(2).unsqueeze(2)
    fake = generator(vectors)

    output = discriminator(fake).view(-1)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()

    optimizerG.step()
    optimizerE.step()

    return D_G_z2


def validation(test_loader, encoder, generator, discriminator, epoch, iter):
    D_x, D_G_z1, D_G_z2 = 0, 0, 0
    for i, data in enumerate(test_loader):
        real_img = data[0].to(device)
        seqs = data[1].to(device)
        lengths = data[2].to(device)
        _, vectors = encoder(seqs, lengths)
        vectors = vectors.unsqueeze(2).unsqueeze(2)

        discriminator.zero_grad()
        output = discriminator(real_img).view(-1)
        D_x += output.mean().item()

        fake = generator(vectors)
        output = discriminator(fake.detach()).view(-1)
        D_G_z1 += output.mean().item()

        generator.zero_grad()
        output = discriminator(fake).view(-1)
        D_G_z2 += output.mean().item()

        validation_contrast_save(real_img, fake, epoch, iter)

    print(
        F"[Validation] -- "
        F"D_x: {D_x / len(test_loader)}; "
        F"D_G_z1: {D_G_z1 / len(test_loader)}; "
        F"D_G_z2: {D_G_z2 / len(test_loader)};"
    )


def epoch_validation(test_loader, encoder, generator, discriminator, epoch):
    D_x, D_G_z1, D_G_z2 = 0, 0, 0
    for i, data in enumerate(test_loader):
        real_img = data[0].to(device)
        seqs = data[1].to(device)
        lengths = data[2].to(device)
        _, vectors = encoder(seqs, lengths)
        vectors = vectors.unsqueeze(2).unsqueeze(2)

        discriminator.zero_grad()
        output = discriminator(real_img).view(-1)
        D_x += output.mean().item()

        fake = generator(vectors)
        output = discriminator(fake.detach()).view(-1)
        D_G_z1 += output.mean().item()

        generator.zero_grad()
        output = discriminator(fake).view(-1)
        D_G_z2 += output.mean().item()

        validation_contrast_save(real_img, fake, epoch)

    print(
        F"[Validation] -- "
        F"D_x: {D_x / len(test_loader)}; "
        F"D_G_z1: {D_G_z1 / len(test_loader)}; "
        F"D_G_z2: {D_G_z2 / len(test_loader)};"
    )


def train(train_loader, test_loader, encoder, generator, discriminator, config):
    num_epochs = config['train']['num_epochs']
    num_iter_print_loss = config['train']['num_iter_print_loss']
    num_iter_validation = config['train']['num_iter_validation']

    lr = config['train']['lr']

    optimizerD = optim.Adam(discriminator.parameters(), lr=lr)
    optimizerG = optim.Adam(generator.parameters(), lr=lr)
    optimizerE = optim.Adam(encoder.parameters(), lr=lr)

    criterion = torch.nn.BCELoss()

    D_xs, D_G_z1s, D_G_z2s = [], [], []

    for epoch in range(num_epochs):
        print(F"Begin Epoch[{epoch}]")
        D_x, D_G_z1, D_G_z2 = 0, 0, 0
        D_x_e, D_G_z1_e, D_G_z2_e = 0, 0, 0

        update_D, update_G = 1, 1
        len_epoch = len(train_loader)

        for i, data in tqdm(enumerate(train_loader)):
            if update_D:
                _D_x, _D_G_z1 = one_iter_discriminator(data, encoder, generator, discriminator, optimizerD, criterion)
            else:
                # print('Not updating Discriminator')
                with torch.no_grad():
                    _D_x, _D_G_z1 = one_iter_discriminator(data, encoder, generator, discriminator, optimizerD, criterion, 1)
            _D_G_z2 = one_iter_generator(data, encoder, generator, discriminator, optimizerE, optimizerG, criterion)

            D_x += _D_x
            D_G_z1 += _D_G_z1
            D_G_z2 += _D_G_z2

            D_x_e += _D_x
            D_G_z1_e += _D_G_z1
            D_G_z2_e += _D_G_z2

            if i and i % num_iter_print_loss == 0:
                print(
                    F"[{epoch}][{i}/{len(train_loader)}] -- "
                    F"D_x: {D_x / num_iter_print_loss}; "
                    F"D_G_z1: {D_G_z1 / num_iter_print_loss}; "
                    F"D_G_z2: {D_G_z2 / num_iter_print_loss};"
                )
                update_D = (D_x / num_iter_print_loss < 0.8) or (D_G_z1 / num_iter_print_loss > 0.2)
                D_x, D_G_z1, D_G_z2 = 0, 0, 0

            # if i and i % num_iter_validation == 0:
            #     print("Starting Validation ...")
            #     validation(test_loader, encoder, generator, discriminator, epoch, i)

        epoch_validation(test_loader, encoder, generator, discriminator, epoch)
        D_xs.append(D_x_e / len_epoch)
        D_G_z1s.append(D_G_z1_e / len_epoch)
        D_G_z2s.append(D_G_z2_e / len_epoch)
        save_model(epoch, encoder, generator, discriminator, config['file'])
        plot_zag(D_xs, D_G_z1s, D_G_z2s, config['file'])


if __name__ == '__main__':
    config = load_json(CONFIG_FILE)
    train_loader, test_loader = diloader.get_loader(config)
    encoder = TextEncoder(config['text'])
    encoder = encoder.to(device)
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    weights_init(generator)
    weights_init(discriminator)
    train(train_loader, test_loader, encoder, generator, discriminator, config)
