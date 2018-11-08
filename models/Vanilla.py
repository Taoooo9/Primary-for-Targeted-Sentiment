import torch
import torch.nn as nn
from .Attention import Attention
import torch.nn.functional as F

class Vanilla(nn.Module):

    def __init__(self, args, embedding):
        super(Vanilla, self).__init__()
        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_size, dropout=args.dropout_rnn, bidirectional=True)
        self.attention = Attention(args)
        self.dropout = nn.Dropout(args.dropout_rnn)
        self.w = nn.Linear(args.embed_dim, 3, bias=True)
        torch.nn.init.xavier_uniform(self.w.weight)


    def forward(self, x, start, end, length):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.attention(x, start, end, length)
        logit = F.softmax(self.w(x))
        return logit







