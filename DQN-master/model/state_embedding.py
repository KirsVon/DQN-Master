#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：state_embedding.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/7/7 10:03 
'''

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class my_config():
    max_length = 20
    batch_size = 64
    embedding_size = 256
    hidden_size = 128
    num_layers = 2
    dropout = 0.5
    output_size = 2
    lr = 0.001
    epoch = 5


class stateRefactor(nn.Module):
    def __init__(self, init_size, config: my_config):
        super(stateRefactor, self).__init__()
        self.init_size = init_size
        self.config = config
        self.embeddings = nn.Embedding(init_size, self.config.embedding_size)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=False,  # 是否构建双向LSTM
            batch_first=True)
        self.linear = nn.Linear(self.config.hidden_size, self.config.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq[0], input_seq[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, (h_n, c_n) = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        pred = self.linear(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return h_n
