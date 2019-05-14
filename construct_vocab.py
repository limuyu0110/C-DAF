# -*- coding:utf-8 _*-
""" 
@author:limuyu
@file: construct_vocab.py 
@time: 2019/05/14
@contact: limuyu0110@pku.edu.cn

"""

from config import CONFIG_FILE
from utils import load_json, dump_json
from collections import defaultdict
import re
import torch
from gensim.models import Word2Vec

if __name__ == '__main__':
    config = load_json(CONFIG_FILE)
    text = load_json(config['file']['desc_raw_dir'])
    new_text = {}
    vocab = {'<S>': 0, '<E>': 1, '<UNK>': 2}
    wv_dim = config['text']['embedding']
    vocab_threshold = config['text']['threshold']

    w2v = Word2Vec.load(config['file']['w2v_path'])

    wc = defaultdict(lambda: 0)
    for k, v in text.items():
        words = re.split('\s+', v)
        for word in words:
            wc[word] += 1

    # print(sorted(list(wc.items()), key=lambda x: x[1])[-50:])

    tl = [torch.randn(wv_dim), torch.randn(wv_dim), torch.randn(wv_dim)]
    cnt = 3
    for k, v in wc.items():
        if v > vocab_threshold:
            if k in w2v.wv:
                tl.append(torch.Tensor(w2v.wv[k]))
            else:
                tl.append(torch.randn(wv_dim))
            vocab[k] = cnt
            cnt += 1

    for k, v in text.items():
        words = re.split(r'\s+', v)
        words = list(map(lambda x: '<UNK>' if wc[x] <= vocab_threshold else x, words))[:config['text']['max_length']]
        words = ["<S>"] + words + ["<E>"]
        new_text[k] = ' '.join(words)

    dump_json(vocab, config['file']['vocab_path'])
    dump_json(new_text, config['file']['desc_dir'])
    t = torch.stack(tl)
    torch.save(t, config['text']['embedding_path'])
