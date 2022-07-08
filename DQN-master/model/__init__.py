import torch
from torch import nn

#建立词向量层
embedding = nn.Embedding(10, 3)

input = torch.LongTensor([1,2,4,5],[4,3,2,9])