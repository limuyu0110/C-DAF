# -*- coding:utf-8 _*-
""" 
@author:limuyu
@file: data_loader.py 
@time: 2019/05/13
@contact: limuyu0110@pku.edu.cn

"""

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch
import os
from overrides import overrides
from PIL import Image
import numpy as np

from gensim.models import Word2Vec
import re

from typing import Dict

import sys
sys.path.append('../')
from utils import load_json

my_transform = Compose([
    # Resize((64, 64)),
    Resize((128, 128)),
    ToTensor(),
    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class DIData(Dataset):
    @overrides
    def __init__(self, IMG_folder, TEXT_file, VOCAB_file):
        self.IMG_folder = IMG_folder
        self.fl = [os.path.join(IMG_folder, fn) for fn in os.listdir(IMG_folder)]
        self.tdict = self.load_file(TEXT_file)
        self.w2i = load_json(VOCAB_file)

    @overrides
    def __getitem__(self, idx):
        file = self.fl[idx]
        text_idx = file.split('\\')[-1].replace('.jpg', '')
        tmp = Image.open(file).convert('RGB')

        text_li = list(map(lambda x: self.w2i[x], re.split('[\s+]', self.tdict[text_idx])))

        return my_transform(tmp), text_li

    @overrides
    def __len__(self):
        return len(self.fl)

    @staticmethod
    def load_file(fn) -> Dict[str, str]:
        import codecs
        import json
        with codecs.open(fn, 'r', 'utf8') as f:
            return json.load(f)

    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        imgs, sequences = zip(*data)

        lengths = [len(seq) for seq in sequences]
        np_li = np.array([
            np.array([
                seq[i] if i < len(seq) else 2 for i in range(max(lengths))
            ])
            for seq in sequences
        ])
        return torch.stack(imgs), torch.LongTensor(np_li), torch.LongTensor(lengths)


def get_loader(config):
    batch_size = config['train']['batch_size']
    train_dataset = DIData(config['file']['train_img_dir'], config['file']['desc_dir'], config['file']['vocab_path'])
    test_dataset = DIData(config['file']['test_img_dir'], config['file']['desc_dir'], config['file']['vocab_path'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=test_dataset.collate_fn)
    return train_loader, test_loader
