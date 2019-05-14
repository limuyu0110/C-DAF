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


def validation(test_loader, encoder, generator, discriminator):
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

        validation_contrast(real_img, fake)

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

    for epoch in range(num_epochs):
        print(F"Begin Epoch[{epoch}]")
        errD, errG, D_x, D_G_z1, D_G_z2 = 0, 0, 0, 0, 0
        for i, data in tqdm(enumerate(train_loader)):
            _errD, _errG, _D_x, _D_G_z1, _D_G_z2 = \
                one_iter(data, encoder, generator, discriminator, optimizerE, optimizerG, optimizerD, criterion)
            # accumulating parameters
            errD += _errD
            errG += _errG
            D_x += _D_x
            D_G_z1 += _D_G_z1
            D_G_z2 += _D_G_z2

            if i % num_iter_print_loss == 0:
                print(
                    F"[{epoch}][{i}/{len(train_loader)}] -- "
                    F"errD: {errD / num_iter_print_loss}; "
                    F"errG: {errG / num_iter_print_loss}; "
                    F"D_x: {D_x / num_iter_print_loss}; "
                    F"D_G_z1: {D_G_z1 / num_iter_print_loss}; "
                    F"D_G_z2: {D_G_z2 / num_iter_print_loss};"
                )
                errD, errG, D_x, D_G_z1, D_G_z2 = 0, 0, 0, 0, 0

            if i % num_iter_validation == 0:
                print("Starting Validation ...")
                validation(test_loader, encoder, generator, discriminator)


if __name__ == '__main__':
    config = load_json(CONFIG_FILE)
    train_loader, test_loader = diloader.get_loader(config)
    encoder = TextEncoder(config['text']).to(device)
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    weights_init(generator)
    weights_init(discriminator)
    train(train_loader, test_loader, encoder, generator, discriminator, config)
