#-*- coding:utf-8 _*-  
""" 
@author:limuyu
@file: text_model.py 
@time: 2019/05/13
@contact: limuyu0110@pku.edu.cn

"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.n_layers = config['layers']
        self.embedding = nn.Embedding.from_pretrained(torch.load(config["embedding_path"]), freeze=False)
        self.hidden_size = config['hidden']
        self.bidirectional = config['bidirectional']
        self.dropout = config['dropout']
        self.gru = nn.GRU(config['embedding'], self.hidden_size, self.n_layers,
                          dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input, input_lengths, hidden=None):
        embedded = self.embedding(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # this 'forward' serves as the Seq2Vec Function
        # So we only need the last hidden state
        # the last hidden state dim is (num_layers * direction) * batch * dim
        # hidden should have sum over the first dimension
        return outputs, hidden.sum(dim=0)



