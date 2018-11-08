import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import reduce
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.w1 = nn.Linear(args.embed_dim * 2, args.attention_size, bias=False)
        nn.init.xavier_uniform(self.w1.weight)
        self.u = nn.Linear(args.attention_size, 1, bias=False)
        nn.init.xavier_uniform(self.u.weight)


    def forward(self, x, start, end, length):
        number = end - start + 1
        ht = torch.zeros((number, 300), dtype=torch.float, requires_grad=False)
        ht.copy_(x[start:end+1][0])
        ht = torch.mean(ht, 0)
        ht = ht.unsqueeze(0)
        ht = ht.expand_as(x)
        hi = torch.cat((x, ht), 1)
        h = self.w1(hi)
        h = F.tanh(h)
        beta = self.u(h)
        beta = beta.transpose(0, 1)
        alpha = F.softmax(beta)
        alpha = alpha.transpose(0, 1)
        alpha = alpha.expand(length, 300)
        s = torch.mul(alpha, x)
        s = torch.sum(s, 0, keepdim=True)
        return s











